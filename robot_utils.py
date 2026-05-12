import cv2
import numpy as np
import os

def read_image_files(colour_file):
    cap = cv2.VideoCapture(colour_file)
    colour_frames = []
    while True:
        ret, colour_frame = cap.read()
        if not ret:
            break
        colour_frames.append(colour_frame)
    cap.release()
    print(f"Read {len(colour_frames)} color frames from {colour_file}")
    return colour_frames

#Crop frame to lower half(50%)
def crop_frame(frame):
    h, w = frame.shape[:2]
    return frame[h//2:, :]

#detect yellow colour in given frame and returns a bnary mask
def detect_yellow(colour_frame):
    # Work directly on the passed frame — caller handles cropping
    hsv = cv2.cvtColor(colour_frame, cv2.COLOR_BGR2HSV)

    # thresholds for neon lime-yellow rope:
    # H 20-35 covers yellow through lime-green in OpenCV's 0-179 scale
    # S 100-255 avoids detecting near-grey floor surfaces
    # V 80-255 catches shadowed/textured parts of the braided rope
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    roi_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Fill gaps from braided rope texture and thicken the detection
    kernel = np.ones((7, 7), np.uint8)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
    roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)

    return roi_mask

#FIT A LINEAR LINE TO THE DETECTED YELLOW MASK
def fit_line(mask):
    # Find largest connected component only
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    
    if num_labels < 2:  # only background
        return None
    
    # Get largest non-background component
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_mask = np.uint8(labels == largest) * 255

    # np.where returns (rows, cols) = (y, x) — need to swap for fitLine
    yx = np.column_stack(np.where(clean_mask > 0))
    
    if len(yx) < 10:  # need enough points for a reliable fit
        return None
    
    # Swap to (x, y) format for fitLine
    points = np.column_stack((yx[:, 1], yx[:, 0])).astype(np.float32)
    
    output = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # cv2.fitLine returns shape (4,1) — flatten to scalars
    vx, vy, x0, y0 = output.flatten()
    
    return (float(vx), float(vy), float(x0), float(y0))

def get_line_direction(line, mask, frame_shape,
                       grad_threshold=0.10,
                       centre_threshold=0.15):
    """For dataset labelling and coarse direction."""
    
    pixel_count = int(np.sum(mask > 0))
    if line is None or pixel_count < 1500:
        return None, 0, pixel_count

    vx, vy, x0, y0 = line
    if abs(vy) < 1e-6:
        return 'straight', 0, pixel_count

    gradient    = -float(vx / vy)
    w           = frame_shape[1]
    norm_offset = (float(x0) - w // 2) / (w // 2)

    # Weighted score — offset dominates since it's where robot IS
    score = 0.4 * gradient + 0.6 * norm_offset

    if score < -centre_threshold:
        return 'left', score, pixel_count
    elif score > centre_threshold:
        return 'right', score, pixel_count
    else:
        return 'straight', score, pixel_count

def get_steering_magnitude(line, frame_shape):
    if line is None:
        return 0.0
    
    '''
    use points returned by line fit to calculate 
    a continuous steering magnitude in range [-1, 1]
    '''
    vx, vy, x0, y0 = line
    w = frame_shape[1]

    # avoids division by zero for horizontal lines — treat as straight
    if abs(vy) < 1e-6:
        return 0.0
    
    #negate to match camera orientation — positive=left, negative=right
    gradient    = -float(vx / vy)
    norm_offset = (float(x0) - w // 2) / (w // 2)
    score       = 0.4 * gradient + 0.6 * norm_offset
    return float(np.clip(score, -1.0, 1.0))


def get_line_direction(line, mask, frame_shape,
                       grad_threshold=0.15):
    
    #count pixels in mask to filter out unreliable line fits
    pixel_count = int(np.sum(mask > 0))

    if line is None or pixel_count < 1500:
        return None, 0, pixel_count
    
    #line parameters and frame dimensions
    vx, vy, x0, y0 = line
    if abs(vy) < 1e-6:
        return 'straight', 0, pixel_count
    
    gradient    = -float(vx / vy)
    w           = frame_shape[1]
    #combine gradient and horizontal offset into a single score for direction
    norm_offset = (float(x0) - w // 2) / (w // 2)
    score       = 0.4 * gradient + 0.6 * norm_offset
    if score < -grad_threshold:
        return 'left', score, pixel_count
    elif score > grad_threshold:
        return 'right', score, pixel_count
    else:
        return 'straight', score, pixel_count
    
def get_steering_error(line, frame_shape, lookahead_fraction=0.4):
    if line is None:
        return 0, None
    
    #get line parameters and frame dimensions
    vx, vy, x0, y0 = line
    h, w = frame_shape[:2]
    
    if abs(vy) < 1e-6:
        return 0, None
    
    # Evaluate line at lookahead y position
    y_ahead = int(h * (1.0 - lookahead_fraction))
    x_ahead = int(x0 + (y_ahead - y0) * (vx / vy))
    x_ahead = np.clip(x_ahead, 0, w)
    
    # Negate to match camera orientation — positive=left, negative=right
    error = -(x_ahead - w // 2)
    return float(error), x_ahead

def is_rope_visible(mask, min_continuous_length=60):
    """
    Check rope is continuous and not occluded by feet/objects.
    Uses height of largest connected component as proxy for continuity.
    A foot standing on rope breaks it into short disconnected segments.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels < 2:
        return False

    # Get largest non-background component
    areas       = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + np.argmax(areas)

    # Height of largest component — rope should span vertically
    component_height = stats[largest_idx, cv2.CC_STAT_HEIGHT]

    if component_height < min_continuous_length:
        return False

    return True

def extract_from_frames(frames, output_dir, label_names,
                        frame_skip=3, min_pixels=1000):
    
    '''Extract and label frames based on line fit direction, saving to disk.
    Applies cropping, yellow detection, line fitting, and direction scoring.
    Only saves frames where rope is visible and has enough pixels.
    Used to train ResNet based on labellings'''
    for name in label_names:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    records   = []
    saved     = 0
    frame_idx = 0

    for i, frame in enumerate(frames):
        if i % frame_skip != 0:
            continue

        cropped     = crop_frame(frame)
        mask        = detect_yellow(cropped)
        pixel_count = int(np.sum(mask > 0))

        if pixel_count < min_pixels:
            continue

        # Filter occluded frames
        if not is_rope_visible(mask, min_continuous_length=60):
            continue

        line = fit_line(mask)
        if line is None:
            continue

        direction, score, _ = get_line_direction(line, mask, cropped.shape)
        if direction is None:
            continue

        filename = f"{saved:06d}.jpg"
        filepath = os.path.join(output_dir, direction, filename)
        cv2.imwrite(filepath, frame)
        records.append((filename, direction,
                        label_names.index(direction)))
        saved += 1
        frame_idx += 1

    print(f"Saved: {saved} frames")
    return records

def get_rope_side(mask):
    # Determine if rope is predominantly on left or right half of the frame
    #uses pixel count to deduce
    w = mask.shape[1]

    left_pixels  = int(np.sum(mask[:, :w//2] > 0))
    right_pixels = int(np.sum(mask[:, w//2:] > 0))
    if right_pixels > left_pixels * 1.5:
        return 'right'
    elif left_pixels > right_pixels * 1.5:
        return 'left'
    return None