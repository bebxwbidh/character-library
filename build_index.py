"""人物资产库建库脚本（与场景库 build_index.py 同套路）。

用法：
    python build_index.py [源目录] [--force]

功能：
    1. 扫描源目录（默认 D:/人物资产）的图片（png/jpg）
    2. 按 2026-11-09 建库定规压缩：characters/<性别目录>/<id>.jpg 长边2048 q90（生图参考够用），
       thumbs/<id>.jpg 长边480 q85（目检/挑人用）；4K+ 原图留源目录当母版，不入库
    3. facexlib 自动检测人脸：数量、最大脸像素、置信度、画面位置 → 换算锚点等级
       A=最大脸高≥600px 可直接当生图锚点；B=300~599 可用（建议配合修复）；
       C=<300px 只作妆造/服装参考，直接当锚点会糊
    4. 人工标注表 LABELS 维护性别/妆造/景别/同人组等语义字段
    5. 输出 index.yaml（供模型按字段筛选）
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SRC_DEFAULT = "D:/人物资产"
LIB = Path(__file__).resolve().parent
CHAR_DIR = LIB / "characters"
THUMB_DIR = LIB / "thumbs"

# ---- 人工语义标注（按需增改，id 按源文件名排序 C01 起）----
# gender/shot/hair/outfit/light/group/note；group=同人组（同组任选其一即可，脸一致）
LABELS = {
    "C01": dict(gender="女", shot="半身站姿", hair="黑长直", outfit="黑色圆领T恤居家装", light="室内窗光", group="", note="全身可见身形比例，适合当锚点底图"),
    "C02": dict(gender="女", shot="面部特写", hair="黑长直中分", outfit="露肩白衫", light="柔光棚拍", group="g-f01", note="清冷御姐脸"),
    "C03": dict(gender="女", shot="面部特写", hair="黑长直中分", outfit="白色上衣", light="柔光棚拍", group="g-f01", note="与C02同人连拍，角度稍侧"),
    "C04": dict(gender="男", shot="面部特写", hair="黑色碎盖短发", outfit="深色上衣", light="柔光棚拍", group="", note="少年感清秀脸"),
    "C05": dict(gender="女", shot="面部特写", hair="黑长发低马尾", outfit="浅色上衣", light="柔光棚拍", group="", note="温婉圆脸"),
    "C06": dict(gender="女", shot="面部特写", hair="黑长直", outfit="深色上衣", light="柔光棚拍", group="", note="颧骨立体冷脸"),
    "C07": dict(gender="女", shot="面部特写", hair="黑长发空气刘海", outfit="浅色上衣", light="柔光棚拍", group="", note="甜妹脸"),
    "C08": dict(gender="男", shot="面部特写", hair="黑色短发", outfit="深色上衣", light="柔光棚拍", group="", note="浓眉方脸英气"),
    "C09": dict(gender="男", shot="面部特写", hair="碎盖短发", outfit="灰绿色上衣", light="柔光棚拍", group="", note="邻家少年脸"),
    "C10": dict(gender="女", shot="面部特写", hair="黑色波波头短发", outfit="白色上衣", light="柔光棚拍", group="", note="短发精灵感"),
    "C11": dict(gender="女", shot="面部特写", hair="黑长发", outfit="浅色上衣", light="柔光棚拍", group="", note="桃花眼圆脸"),
    "C12": dict(gender="男", shot="面部特写", hair="黑色短碎发", outfit="深色上衣", light="柔光棚拍", group="", note="剑眉薄唇俊朗"),
    "C13": dict(gender="男", shot="面部特写", hair="寸头/短背头", outfit="深色上衣", light="柔光棚拍", group="", note="硬朗成熟脸"),
    "C14": dict(gender="女", shot="面部特写", hair="黑长发中分", outfit="浅色上衣", light="柔光棚拍", group="", note="古典温婉脸"),
    "C15": dict(gender="男", shot="面部特写", hair="黑色短发", outfit="白衬衫", light="柔光棚拍", group="", note="浓颜系男主角脸"),
    "C16": dict(gender="女", shot="正面定妆半身", hair="中式双花苞头+头饰", outfit="新娘妆造（中式）", light="柔和影棚光", group="", note="中式新娘定妆，妆造参考佳品"),
    "C17": dict(gender="女", shot="半身人像", hair="黑长发", outfit="深色上衣", light="室内柔光暗背景", group="", note="静谧氛围人像"),
    "C18": dict(gender="女", shot="白纱新娘半身", hair="盘发+白纱头纱", outfit="白色婚纱", light="柔光影棚", group="g-bride1", note="西式新娘定妆半身"),
    "C19": dict(gender="女", shot="白纱新娘特写", hair="盘发+白纱头纱", outfit="白色婚纱", light="柔光影棚", group="g-bride1", note="与C18同人，脸更近"),
    "C20": dict(gender="女", shot="面部特写", hair="赤褐发长卷发", outfit="深色上衣", light="柔光棚拍", group="", note="红棕发异色系"),
    "C21": dict(gender="女", shot="半身人像", hair="黑长发", outfit="白衬衫+深色西装职场装", light="室内自然光", group="", note="清冷职场风"),
    "C22": dict(gender="女", shot="面部特写", hair="黑长发", outfit="浅色上衣", light="自然暖光", group="", note="2026-11-08 套系女锚点原始素材"),
    "C23": dict(gender="女", shot="面部特写", hair="黑长发", outfit="深色上衣", light="自然暖光", group="", note="暖调自然光人像"),
    "C24": dict(gender="女", shot="面部特写", hair="黑长发", outfit="深色上衣", light="冷调自然光", group="", note="冷调东方少女脸"),
}


def face_grade(hpx: int) -> str:
    if hpx >= 600:
        return "A"
    if hpx >= 300:
        return "B"
    return "C"


def detect_faces(img_bgr):
    """facexlib CPU 检测；返回 (数量, 最大脸高px, 置信度, 相对位置x,y)。失败返回 None。"""
    try:
        import cv2
        import numpy as np
        import torch
        from facexlib.detection import init_detection_model

        global _DET
        try:
            det = _DET
        except NameError:
            home = Path(os.environ.get("GFPGAN_HOME", Path.home() / ".workbuddy" / "models" / "gfpgan"))
            os.chdir(home)
            det = init_detection_model("retinaface_resnet50", half=False, device="cpu",
                                       model_rootpath=str(home / "gfpgan"))
            _DET = det
        with torch.no_grad():
            r = det.detect_faces(img_bgr)
        arr = np.asarray(r[0] if isinstance(r, tuple) else r)
        if arr.size == 0:
            return 0, 0, 0.0, None, None
        arr = arr[arr[:, 4] > 0.5][:, :5]
        h_img, w_img = img_bgr.shape[:2]
        x1, y1, x2, y2, s = max(arr, key=lambda b: b[4])
        h_img_idx = 1 if img_bgr.ndim == 3 else 0
        return (len(arr), int(y2 - y1), float(s),
                round(float((x1 + x2) / 2 / w_img), 2),
                round(float((y1 + y2) / 2 / h_img), 2))
    except Exception as e:
        print(f"  [检测失败] {e}")
        return None


def compress(src: Path, dst: Path, long_edge: int, quality: int):
    im = Image.open(src).convert("RGB")
    im.thumbnail((long_edge, long_edge), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=quality)
    return im.size


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else Path(SRC_DEFAULT)
    files = sorted([f for f in os.listdir(src)
                    if f.lower().endswith((".png", ".jpg", ".jpeg")) and (src / f).is_file()])
    print(f"[源] {src} 共 {len(files)} 张")

    import cv2

    entries = []
    for i, f in enumerate(files, 1):
        cid = f"C{i:02d}"
        lab = LABELS.get(cid, {})
        gender = lab.get("gender")
        char = CHAR_DIR / f"{gender}生" / f"{cid}.jpg" if gender in ("男", "女") else CHAR_DIR / cid / "char.jpg"
        thumb = THUMB_DIR / f"{cid}.jpg"
        size_c = compress(src / f, char, 2048, 90)
        compress(src / f, thumb, 480, 85)

        # cv2 中文路径安全读
        img = cv2.imdecode(np.fromfile(str(char), dtype=np.uint8), 1)
        det = detect_faces(img)
        if det:
            n, hpx, conf, rx, ry = det
            grade = face_grade(hpx) if n else "-"
        else:
            n, hpx, conf, rx, ry, grade = 0, 0, 0.0, None, None, "-"

        entries.append(dict(
            id=cid, file=char.relative_to(CHAR_DIR.parent).as_posix(), source=f, gender=lab.get("gender", "?"), shot=lab.get("shot", ""),
            hair=lab.get("hair", ""), outfit=lab.get("outfit", ""), light=lab.get("light", ""),
            group=lab.get("group", ""), note=lab.get("note", ""),
            size=f"{size_c[0]}x{size_c[1]}", faces=n, face_hpx=hpx,
            face_conf=round(conf, 2), face_pos=f"x{rx} y{ry}" if rx is not None else "",
            anchor=grade,
        ))
        print(f"[{cid}] {f[:28]:<28} 脸{hpx}px {grade} 级")

    # 写 index.yaml
    lines = ["# 人物资产库索引（build_index.py 自动生成 + LABELS 人工标注）",
             "# anchor 等级：A=脸>=600px 可直接当锚点；B=300~599 可用；C=<300px 只作妆造/服装参考",
             "library: character-library",
             f"updated: 2026-11-09",
             f"count: {len(entries)}",
             "characters:"]
    for e in entries:
        lines.append(f"  - id: {e['id']}")
        lines.append(f"    file: {e['file']}")
        lines.append(f"    thumb: thumbs/{e['id']}.jpg")
        lines.append(f"    source: \"{e['source']}\"")
        lines.append(f"    gender: {e['gender']}")
        lines.append(f"    shot: {e['shot']}")
        lines.append(f"    hair: {e['hair']}")
        lines.append(f"    outfit: \"{e['outfit']}\"")
        lines.append(f"    light: {e['light']}")
        if e["group"]:
            lines.append(f"    group: {e['group']}")
        lines.append(f"    size: {e['size']}")
        lines.append(f"    faces: {e['faces']}")
        lines.append(f"    face_hpx: {e['face_hpx']}")
        lines.append(f"    face_conf: {e['face_conf']}")
        if e["face_pos"]:
            lines.append(f"    face_pos: \"{e['face_pos']}\"")
        lines.append(f"    anchor: {e['anchor']}")
        lines.append(f"    note: \"{e['note']}\"")
    (LIB / "index.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[索引] index.yaml 写入 {len(entries)} 条")


if __name__ == "__main__":
    main()
