import cv2
import numpy as np

def process_bubble_image_robust(image_path, output_path="result.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        print("图片读取失败")
        return 0, None

    # 统一缩放到可控尺寸，后续阈值和面积参数更稳定。
    img = cv2.resize(img, None, fx=0.25, fy=0.25)
    img_result = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1) 自动找圆盘样本区域：选取靠下的候选圆，避免固定矩形 ROI 带来的偏差。
    blur_for_circle = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blur_for_circle,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40, h // 8),
        param1=120,
        param2=35,
        minRadius=max(40, int(min(h, w) * 0.10)),
        maxRadius=int(min(h, w) * 0.48),
    )

    if circles is None:
        print("未检测到样本圆盘，请检查光照或拍摄角度")
        return 0, img_result

    circles = np.round(circles[0]).astype(int)
    # 优先选择位置靠下、半径较大的圆盘作为样本。
    sample_circle = max(circles, key=lambda c: (c[1], c[2]))
    cx, cy, r = sample_circle.tolist()

    mask = np.zeros_like(gray)
    cv2.circle(mask, (cx, cy), int(r * 0.92), 255, -1)

    # 2) 孔洞增强：CLAHE + 黑帽，突出“亮背景上的小暗孔”。
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bubble_enhanced = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, blackhat_kernel)

    # 3) 仅在圆盘内阈值分割，并清理孤立噪点。
    bubble_enhanced = cv2.bitwise_and(bubble_enhanced, bubble_enhanced, mask=mask)
    binary = cv2.adaptiveThreshold(
        bubble_enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        -2,
    )
    binary = cv2.bitwise_and(binary, binary, mask=mask)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_kernel)

    # 4) 轮廓过滤：面积 + 圆度，剔除划痕/边缘碎片。
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_bubbles = []

    min_area = max(3, int(r * r * 0.00003))
    max_area = max(min_area + 1, int(r * r * 0.003))
    inner_r2 = (r * 0.88) * (r * 0.88)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.35:
            continue

        m = cv2.moments(c)
        if m["m00"] == 0:
            continue

        ccx = m["m10"] / m["m00"]
        ccy = m["m01"] / m["m00"]
        if (ccx - cx) * (ccx - cx) + (ccy - cy) * (ccy - cy) > inner_r2:
            continue

        valid_bubbles.append(c)

    # 5) 可视化输出。
    cv2.circle(img_result, (cx, cy), r, (255, 0, 0), 2)
    cv2.drawContours(img_result, valid_bubbles, -1, (0, 255, 0), 1)
    cv2.putText(
        img_result,
        f"bubbles: {len(valid_bubbles)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if output_path:
        cv2.imwrite(output_path, img_result)

    return len(valid_bubbles), img_result

if __name__ == "__main__":
    count, result_img = process_bubble_image_robust("buble.jpg", "result.jpg")
    print(f"检测到的有效气泡数: {count}")
    if result_img is not None:
        print("已保存可视化结果到 result.jpg")