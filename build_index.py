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
# gender/shot/hair/outfit/light/group/note/stype；group=同人组（同组任选其一即可，脸一致）
# stype = 质感来源：AI生成（模型生图，会外溢 CG 质感与棚拍妆面，case-2026-11-18-06）| 实拍
#         交付严质感批优先选「实拍」锚点；缺省视为 AI生成
LABELS = {
    "C01": dict(gender="男", shot="四视图定妆板", hair="黑发背头", outfit="白色双排扣礼服+白领结+刺绣白裤+白皮鞋", light="灰底棚拍", group="", note="黑裤白鞋正式新郎造型（四视图板，单脸约 380px，未检测）", stype="AI生成"),
    "C02": dict(gender="女", shot="四视图定妆板", hair="黑长直半扎发", outfit="白色一字肩蕾丝鱼尾婚纱+长头纱", light="灰底棚拍", group="", note="白纱新娘定妆，含背面蝴蝶结（四视图板，单脸约 380px，未检测）", stype="AI生成"),
    "C03": dict(gender="女", shot="四视图定妆板", hair="棕长卷发", outfit="黑色蕾丝胸衣连衣短裙+黑丝吊袜带+黑高跟", light="灰底棚拍", group="", note="私房/夜宴风，注意公库可见（四视图板，单脸约 330px，未检测）", stype="AI生成"),
    "C04": dict(gender="男", shot="四视图定妆板", hair="黑色短发", outfit="白色燕尾礼服+粉格纹领带+胸花", light="灰底棚拍", group="", note="白燕尾新郎造型，长款外套（四视图板，单脸约 330px，未检测）", stype="AI生成"),
    "C05": dict(gender="男", shot="四视图定妆板", hair="黑色短发", outfit="黑色双排扣西装+黑衬衫+黑领结", light="灰底棚拍", group="", note="全黑礼服，硬朗新郎（四视图板，单脸约 330px，未检测）", stype="AI生成"),
    "C06": dict(gender="女", shot="四视图定妆板", hair="黑棕披发+缎带发饰", outfit="黑色吊带伞裙+白蕾丝裙边+蕾丝袜+玛丽珍鞋", light="灰底棚拍", group="", note="复古少女/哥特风（四视图板，单脸约 330px，未检测）", stype="AI生成"),
    "C07": dict(gender="女", shot="四视图定妆板", hair="黑长直", outfit="白色抹胸不规则荷叶边长裙", light="灰底棚拍", group="", note="轻婚纱/晚宴白裙（四视图板，单脸约 330px，未检测）", stype="AI生成"),
    "C08": dict(gender="男", shot="四视图定妆板", hair="黑色长发马尾+胡须", outfit="黑色三件套礼服+黑领结", light="灰底棚拍", group="", note="长发胡须气质，艺术感新郎（四视图板，单脸约 330px，未检测）", stype="AI生成"),
    "C09": dict(gender="女", shot="四视图定妆板", hair="黑色盘发", outfit="白色薄纱披肩婚纱+水晶皇冠+珍珠耳坠", light="灰底棚拍", group="", note="皇冠造型，薄纱披肩（四视图板，单脸约 330px，未检测）", stype="AI生成"),
    "C10": dict(gender="女", shot="四视图定妆板", hair="棕色盘发", outfit="白色立体玫瑰短裙+白色高跟鞋", light="灰底棚拍", group="", note="立体玫瑰短裙，俏皮新娘（四视图板，单脸约 330px，未检测）", stype="AI生成"),
    "C11": dict(gender="女", shot="四视图定妆板", hair="黑色盘发", outfit="白色缎面一字肩蓬蓬裙+头纱+金色大耳环", light="灰底棚拍", group="", note="缎面蓬裙，复古新娘（四视图板，单脸约 330px，未检测）", stype="AI生成"),
    "C12": dict(gender="女", shot="四视图定妆板", hair="黑长微卷+墨绿针织帽", outfit="棕格纹衬衫+白内搭+黑阔腿裤+绿针织帽", light="灰底棚拍", group="", note="休闲复古，日常感造型（四视图板，单脸约 330px，未检测）", stype="AI生成"),
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
            group=lab.get("group", ""), note=lab.get("note", ""), source_type=lab.get("stype", "AI生成"),
            size=f"{size_c[0]}x{size_c[1]}", faces=n, face_hpx=hpx,
            face_conf=round(conf, 2), face_pos=f"x{rx} y{ry}" if rx is not None else "",
            anchor=grade,
        ))
        print(f"[{cid}] {f[:28]:<28} 脸{hpx}px {grade} 级")

    # 写 index.yaml
    lines = ["# 人物资产库索引（build_index.py 自动生成 + LABELS 人工标注）",
             "# anchor 等级：A=脸>=600px 可直接当锚点；B=300~599 可用；C=<300px 只作妆造/服装参考",
             "# source_type：AI生成=模型生图（质感外溢风险，见 case-2026-11-18-06）| 实拍=真实照片；交付严质感批优先实拍锚点",
             "library: character-library",
             f"updated: 2026-11-18",
             f"count: {len(entries)}",
             "characters:"]
    for e in entries:
        lines.append(f"  - id: {e['id']}")
        lines.append(f"    file: {e['file']}")
        lines.append(f"    thumb: thumbs/{e['id']}.jpg")
        lines.append(f"    source: \"{e['source']}\"")
        lines.append(f"    source_type: {e['source_type']}")
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
