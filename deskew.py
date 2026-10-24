#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deskew.py —— 图片倾斜校正（去倾斜），让 PaddleOCR 更容易识别倾斜文字。

原理
----
1. 估计倾斜角，两种方式：
   paper（默认优先）：找出图中高饱和度的彩色纸张（例如本图的橙色书法横幅），
       对纸张像素做主成分分析（PCA）。纸张是长条形，主方向就是文字行方向，
       据此得到需要旋转的角度。适合“彩色纸张 + 文字”的照片。
   projection（兜底）：把二值图按一堆候选角度旋转，取“水平投影方差最大”的角度。
       适合白底黑字的文档、告示牌；照片背景复杂时可能不准。
2. 用扩展画布旋转整张图（空白填白），不会裁掉边角。
3. 若检测到纸张区域，额外裁出该区域并放大（默认 2 倍），得到更适合 OCR 的图。

用法
----
    python deskew.py                                   # 默认处理脚本同目录的 sign_10.jpg
    python deskew.py sign_10.jpg
    python deskew.py sign_10.jpg -o straightened.jpg
    python deskew.py sign_10.jpg --angle 15            # 手动指定角度，正值 = 顺时针
    python deskew.py sign_10.jpg --roi 100 600 980 860 # 先把图片裁剪到该区域再处理
    python deskew.py sign_10.jpg --method projection --no-crop
    python deskew.py sign_10.jpg --debug               # 额外输出检测结果示意图

输出
----
    <文件名>_deskewed.jpg        整图旋转拉正后的结果
    <文件名>_deskewed_crop.jpg   检测到的纸张区域（裁切 + 放大），推荐直接喂给 PaddleOCR
    <文件名>_deskew_debug.jpg    --debug 时输出，绿色为检测到的纸张，并标注角度

角度约定
--------
    --angle 以及脚本打印的角度，均为“顺时针旋转角度”：
    正值表示需要把图片顺时针旋转该角度，负值表示逆时针。

依赖
----
    pip install opencv-python numpy

注意
----
    若你的图片不是彩色纸张（例如普通白底文档），可加 --method projection；
    若自动角度不对，用 --angle 手动指定，先用 --debug 看看检测框是否选中了文字。
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print("缺少 OpenCV，请先安装：pip install opencv-python numpy")
    raise

# Windows 下输出重定向（如管道、IDE）时，统一用 UTF-8，避免中文乱码
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 可调参数
# ---------------------------------------------------------------------------
# 彩色纸张的 HSV 范围（OpenCV: H 0~179, S/V 0~255）。本图的橙色纸张约 H=14~16。
PAPER_HSV_LOW = (5, 150, 120)
PAPER_HSV_HIGH = (32, 255, 255)

# 纸张连通域的最小面积（占整幅图的比例），小于它的色块被忽略，用来排除红色杂物。
PAPER_MIN_AREA_RATIO = 0.005

# 投影法处理前把图缩到最长边不超过该像素，兼顾速度和精度。
PROJ_MAX_SIDE = 900

# 纸张区域长宽比低于该值时认为形状太“方”，PCA 方向不可靠，改用投影法。
ANISO_MIN = 1.5

# 裁剪纸张时四周额外留白（像素）。
CROP_PAD = 20

# 裁剪图默认放大倍数（文字太小会明显影响 PaddleOCR 识别率）。
DEFAULT_SCALE = 2.0
# ---------------------------------------------------------------------------


def imread_unicode(path):
    """读取图片，支持 Windows 下的中文路径。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """保存图片，支持 Windows 下的中文路径。"""
    path = Path(path)
    ext = path.suffix.lower() or ".jpg"
    if ext in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    elif ext == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    else:
        params = []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError("图片编码失败：%s" % path)
    buf.tofile(str(path))


def paper_mask(img):
    """根据颜色找出图中的彩色纸张，返回二值掩膜。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(PAPER_HSV_LOW, dtype=np.uint8),
        np.array(PAPER_HSV_HIGH, dtype=np.uint8),
    )
    # 去噪点、补小洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    h, w = img.shape[:2]
    min_area = max(1500, int(PAPER_MIN_AREA_RATIO * h * w))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def estimate_angle_paper(img):
    """
    用彩色纸张像素的 PCA 主方向估计倾斜角。
    返回 (顺时针角度, 掩膜, 长宽比)；失败时角度为 None。
    """
    mask = paper_mask(img)
    if cv2.countNonZero(mask) == 0:
        return None, None, 0.0

    ys, xs = np.nonzero(mask)
    # 像素太多时抽样，避免大图占用过多内存
    if xs.size > 200000:
        idx = np.linspace(0, xs.size - 1, 200000).astype(np.int64)
        xs, ys = xs[idx], ys[idx]
    pts = np.column_stack((xs, ys)).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    if evals[1] <= 1e-9:
        return None, mask, 0.0
    aniso = float(evals[0] / evals[1])
    if aniso < ANISO_MIN:          # 形状太接近正方形，方向不可信
        return None, mask, aniso

    vx, vy = evecs[:, 0]
    ang_cv = math.degrees(math.atan2(vy, vx))   # 图像坐标下的主方向
    # 归一到 (-90, 90]
    if ang_cv > 90.0:
        ang_cv -= 180.0
    elif ang_cv <= -90.0:
        ang_cv += 180.0
    # cv2 中正角度是逆时针；这里统一换算成“顺时针为正”
    return -ang_cv, mask, aniso


def estimate_angle_projection(img):
    """
    投影法：枚举角度旋转二值图，找水平投影方差最大的角度。
    返回顺时针角度（度）。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = min(1.0, PROJ_MAX_SIDE / float(max(h, w)))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    h, w = gray.shape[:2]

    # 自适应二值化，文字为白色（前景）
    binimg = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    # 去掉长横线 / 长竖线（表格线、建筑轮廓等干扰）
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(w * 0.04)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(9, int(h * 0.04))))
    binimg = cv2.subtract(binimg, cv2.morphologyEx(binimg, cv2.MORPH_OPEN, hk))
    binimg = cv2.subtract(binimg, cv2.morphologyEx(binimg, cv2.MORPH_OPEN, vk))

    ys, xs = np.nonzero(binimg)
    if xs.size < 50:
        return 0.0
    binimg = binimg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hh, ww = binimg.shape
    cx, cy = ww / 2.0, hh / 2.0

    best_score, best_ang = -1.0, 0.0
    for ang in np.arange(-45.0, 45.01, 0.5):
        M = cv2.getRotationMatrix2D((cx, cy), float(ang), 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw = int(hh * sin + ww * cos)
        nh = int(hh * cos + ww * sin)
        M[0, 2] += nw / 2.0 - cx
        M[1, 2] += nh / 2.0 - cy
        r = cv2.warpAffine(
            binimg, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0
        )
        # 按行求和（归一化行宽，避免不同角度裁切范围不同带来的偏差）
        proj = r.sum(axis=1, dtype=np.float64) / max(1.0, float(r.shape[1]))
        score = float(proj.var())
        if score > best_score:
            best_score, best_ang = score, float(ang)

    # 旋转了 best_ang（cv2 逆时针为正）后文字水平，换算成顺时针角度
    return -best_ang


def rotate_expand(img, angle_cw, border_value=(255, 255, 255)):
    """
    按“顺时针 angle_cw 度”旋转图片，并扩大画布避免裁切。
    返回 (旋转后的图, 变换矩阵 M, (画布宽, 画布高))。
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, -angle_cw, 1.0)  # cv2 正角为逆时针
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(math.ceil(h * sin + w * cos))
    nh = int(math.ceil(h * cos + w * sin))
    M[0, 2] += nw / 2.0 - center[0]
    M[1, 2] += nh / 2.0 - center[1]
    rot = cv2.warpAffine(
        img,
        M,
        (nw, nh),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rot, M, (nw, nh)


def crop_paper(rot, mask, M, size, pad=CROP_PAD, scale=DEFAULT_SCALE):
    """在旋转后的图上裁出纸张区域并放大；失败返回 None。"""
    nw, nh = size
    rmask = cv2.warpAffine(mask, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)

    # 黑字在掩膜上是“洞”，先填洞再取外接矩形
    # 外面补一圈 0 再漫水填充，避免纸张贴边时填充结果异常
    padded = cv2.copyMakeBorder(rmask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    fmask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, fmask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    filled = rmask | holes
    filled = cv2.dilate(filled, np.ones((9, 9), np.uint8))

    ys, xs = np.nonzero(filled)
    if xs.size == 0:
        return None
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(nw - 1, int(xs.max()) + pad)
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(nh - 1, int(ys.max()) + pad)
    crop = rot[y1:y2 + 1, x1:x2 + 1]
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None

    if scale and scale > 0 and abs(scale - 1.0) > 1e-3:
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interp)
    return crop


def save_debug(img, mask, angle_cw, method, path):
    """输出调试图：绿色为选中的纸张区域，并标注方法/角度。"""
    dbg = img.copy()
    if mask is not None:
        green = np.zeros_like(dbg)
        green[:] = (0, 255, 0)
        sel = mask > 0
        dbg[sel] = (0.55 * dbg[sel] + 0.45 * green[sel]).astype(np.uint8)
    text = "%s  angle=%+.2f deg (clockwise)" % (method, angle_cw)
    cv2.putText(dbg, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255), 2, cv2.LINE_AA)
    imwrite_unicode(path, dbg)


def build_parser():
    p = argparse.ArgumentParser(
        description="图片倾斜校正（去倾斜），为 PaddleOCR 准备更正的输入图。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", nargs="?", default=None,
                   help="输入图片路径；省略时使用脚本同目录的 sign_10.jpg")
    p.add_argument("-o", "--output", default=None,
                   help="输出图片路径；默认 <输入名>_deskewed.jpg")
    p.add_argument("--angle", type=float, default=None,
                   help="手动指定顺时针旋转角度（度），指定后不再自动检测")
    p.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
                   default=None,
                   help="只处理该矩形区域（像素坐标）；区域内尽量包含完整的纸张，否则角度可能不准")
    p.add_argument("--method", choices=["auto", "paper", "projection"], default="auto",
                   help="角度检测方法：auto=先试彩色纸张再退回投影法")
    p.add_argument("--no-crop", action="store_true",
                   help="不输出纸张区域的裁切放大图")
    p.add_argument("--scale", type=float, default=DEFAULT_SCALE,
                   help="裁切图的放大倍数")
    p.add_argument("--debug", action="store_true",
                   help="额外输出检测掩膜示意图")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    script_dir = Path(__file__).resolve().parent

    in_path = Path(args.input) if args.input else (script_dir / "sign_10.jpg")
    if not in_path.is_absolute() and not in_path.exists():
        # 允许直接写文件名（在别的目录运行脚本时也能找到桌面上的图）
        alt = script_dir / in_path
        if alt.exists():
            in_path = alt
    if not in_path.exists():
        print("找不到输入图片：%s" % in_path)
        return 2

    img = imread_unicode(in_path)
    if img is None:
        print("无法读取图片，请确认格式：%s" % in_path)
        return 2

    if args.roi:
        x1, y1, x2, y2 = args.roi
        h, w = img.shape[:2]
        x1, x2 = sorted((max(0, min(w - 1, x1)), max(0, min(w - 1, x2))))
        y1, y2 = sorted((max(0, min(h - 1, y1)), max(0, min(h - 1, y2))))
        if x2 - x1 < 20 or y2 - y1 < 20:
            print("--roi 区域太小或坐标无效：%s" % (args.roi,))
            return 2
        img = img[y1:y2 + 1, x1:x2 + 1]

    mask = None
    if args.angle is not None:
        angle_cw, method = args.angle, "manual"
    else:
        angle_cw, method = None, None
        if args.method in ("auto", "paper"):
            angle_cw, mask, aniso = estimate_angle_paper(img)
            if angle_cw is not None:
                method = "paper"
            elif args.method == "paper":
                print("未检测到可用的彩色纸张区域（长宽比 %.2f），"
                      "请改用 --angle 手动指定角度。" % aniso)
                return 1
        if angle_cw is None:
            angle_cw = estimate_angle_projection(img)
            method = "projection"
            mask = None   # 角度不是由纸张方向得到的，不按纸张区域裁剪，避免误裁

    rot, M, size = rotate_expand(img, angle_cw)

    # 输出路径
    suffix = in_path.suffix if in_path.suffix else ".jpg"
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = in_path.with_name(in_path.stem + "_deskewed" + suffix)
    imwrite_unicode(out_path, rot)
    print("检测方式 : %s" % method)
    print("倾斜角度 : %+.2f°（正值=顺时针，已按此角度校正）" % angle_cw)
    print("整图输出 : %s" % out_path)

    if mask is not None and not args.no_crop:
        crop = crop_paper(rot, mask, M, size, pad=CROP_PAD, scale=args.scale)
        if crop is not None:
            crop_path = out_path.with_name(out_path.stem + "_crop" + out_path.suffix)
            imwrite_unicode(crop_path, crop)
            print("OCR 裁剪 : %s  (%d x %d)" % (crop_path, crop.shape[1], crop.shape[0]))
        else:
            print("OCR 裁剪 : 跳过（纸张区域为空）")

    if args.debug:
        dbg_path = out_path.with_name(out_path.stem.replace("_deskewed", "") +
                                      "_deskew_debug" + out_path.suffix)
        save_debug(img, mask, angle_cw, method, dbg_path)
        print("调试图   : %s" % dbg_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
