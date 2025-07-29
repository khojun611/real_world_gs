import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import struct
import argparse
from collections import namedtuple

# (이전과 동일한 Camera, Image, Point3D namedtuple 정의)
# ... (qvec2rotmat, read_colmap_bin, generate_sparse_depth_map, create_variance_map 함수는 이전과 동일) ...
# (import 문들 바로 아래에 추가)

# COLMAP 모델 ID와 파라미터 정보 매핑
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), # f, cx, cy
    1: ("PINHOLE", 4),        # fx, fy, cx, cy
    2: ("SIMPLE_RADIAL", 4),  # f, cx, cy, k
    3: ("RADIAL", 5),         # f, cx, cy, k1, k2
    4: ("OPENCV", 8),         # fx, fy, cx, cy, k1, k2, p1, p2
}
Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

def qvec2rotmat(qvec):
    """쿼터니언을 회전 행렬로 변환"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])

def read_colmap_bin(path, bin_type):
    """COLMAP의 .bin 파일을 파싱하는 함수 (최종 수정본)"""
    if bin_type == "cameras":
        with open(path, "rb") as fid:
            num_cameras = struct.unpack('<Q', fid.read(8))[0]
            cameras = {}
            for _ in range(num_cameras):
                cam_id, model_id, width, height = struct.unpack('<iiQQ', fid.read(24))
                
                if model_id not in CAMERA_MODELS:
                    print(f"Warning: Skipping unsupported camera model_id: {model_id}")
                    # 파라미터 부분을 건너뛰기 위해 파일 포인터를 이동시켜야 함
                    # 이 예제에서는 지원하지 않는 모델이 없다고 가정하고 넘어갑니다.
                    continue
                
                model_name, num_params = CAMERA_MODELS[model_id]
                params = struct.unpack('<' + 'd' * num_params, fid.read(8 * num_params))
                
                cameras[cam_id] = Camera(cam_id, model_name, width, height, np.array(params))
            return cameras

    if bin_type == "images":
        with open(path, "rb") as fid:
            num_images = struct.unpack('<Q', fid.read(8))[0]
            images = {}
            for _ in range(num_images):
                image_id = struct.unpack('<I', fid.read(4))[0]
                qvec = np.array(struct.unpack('<dddd', fid.read(32)))
                tvec = np.array(struct.unpack('<ddd', fid.read(24)))
                camera_id = struct.unpack('<I', fid.read(4))[0]
                name = ""
                char = fid.read(1).decode('latin-1')
                while char != '\x00':
                    name += char
                    char = fid.read(1).decode('latin-1')
                
                num_points2D = struct.unpack('<Q', fid.read(8))[0]
                # 깊이맵 생성에 2D 포인트 정보는 필요 없으므로 건너뜀
                fid.seek(num_points2D * 24, 1)
                images[image_id] = Image(image_id, qvec, tvec, camera_id, name, None, None)
            return images

    if bin_type == "points3D":
        with open(path, "rb") as fid:
            num_points = struct.unpack('<Q', fid.read(8))[0]
            points3D = {}
            for _ in range(num_points):
                point_id, = struct.unpack('<Q', fid.read(8))
                xyz = np.array(struct.unpack('<ddd', fid.read(24)))
                rgb = np.array(struct.unpack('<BBB', fid.read(3)))
                error, = struct.unpack('<d', fid.read(8))
                track_len, = struct.unpack('<Q', fid.read(8))
                
                # 깊이맵 생성을 위해 트랙 정보를 읽어야 함
                track = []
                for __ in range(track_len):
                    track.append(struct.unpack('<II', fid.read(8)))
                image_ids = np.array([t[0] for t in track])
                point2D_idxs = np.array([t[1] for t in track])
                points3D[point_id] = Point3D(point_id, xyz, rgb, error, image_ids, point2D_idxs)
            return points3D

    return {}

def generate_sparse_depth_map(image_info, camera_info, points3D):
    """희소 3D 포인트로부터 깊이 맵을 생성"""
    H, W = camera_info.height, camera_info.width
    depth_map = np.zeros((H, W), dtype=np.float32)

    # 월드->카메라 변환 행렬
    R = qvec2rotmat(image_info.qvec)
    t = image_info.tvec
    
    # 3D 포인트를 순회하며 이미지에 투영
    for pt3d in points3D.values():
        # 이 이미지가 해당 3D 포인트를 관측했는지 확인
        if image_info.id in pt3d.image_ids:
            # 월드 좌표를 카메라 좌표로 변환
            p_cam = R @ pt3d.xyz + t
            depth = p_cam[2]

            if depth < 1e-3: continue # 카메라 뒤쪽 포인트는 무시

            # 카메라 내부 파라미터
            fx, fy, cx, cy = 0, 0, 0, 0
            if camera_info.model == "SIMPLE_PINHOLE":
                fx = fy = camera_info.params[0]
                cx, cy = camera_info.params[1], camera_info.params[2]
            elif camera_info.model == "PINHOLE":
                fx, fy = camera_info.params[0], camera_info.params[1]
                cx, cy = camera_info.params[2], camera_info.params[3]
            
            # 픽셀 좌표로 투영
            u = int(round(fx * p_cam[0] / depth + cx))
            v = int(round(fy * p_cam[1] / depth + cy))
            
            if 0 <= u < W and 0 <= v < H:
                # 더 가까운 포인트로 깊이 값 업데이트
                if depth_map[v, u] == 0 or depth_map[v, u] > depth:
                    depth_map[v, u] = depth
    return depth_map

def create_variance_map(
    ref_image: np.ndarray,
    ref_pose: np.ndarray,
    ref_depth: np.ndarray,
    other_images: list[np.ndarray],
    other_poses: list[np.ndarray],
    intrinsics: np.ndarray,
    gamma: float = 1.0
) -> np.ndarray:
    """주어진 수식을 바탕으로 기준 이미지에 대한 분산 맵(reflection score map)을 생성합니다."""
    H, W, _ = ref_image.shape
    all_images = [ref_image] + other_images
    all_poses = [ref_pose] + other_poses
    N = len(all_images)
    variance_map = np.zeros((H, W), dtype=np.float32)

    valid_depth_mask = ref_depth > 0
    v_coords, u_coords = np.where(valid_depth_mask)
    if len(v_coords) == 0:
        print("Warning: No valid depth points found.")
        return variance_map
        
    pixels_2d = np.stack([u_coords, v_coords, np.ones_like(u_coords)], axis=0)
    depth_values = ref_depth[v_coords, u_coords]

    inv_intrinsics = np.linalg.inv(intrinsics)
    points_camera_normalized = inv_intrinsics @ pixels_2d
    points_camera = points_camera_normalized * depth_values
    points_camera_homogeneous = np.vstack([points_camera, np.ones(points_camera.shape[1])])
    inv_ref_pose = np.linalg.inv(ref_pose)
    points_world = (inv_ref_pose @ points_camera_homogeneous).T[:, :3]

    num_valid_pixels = len(u_coords)
    all_colors = np.zeros((N, num_valid_pixels, 3), dtype=np.float32)

    for i, (img, pose) in enumerate(zip(all_images, all_poses)):
        h_img, w_img, _ = img.shape
        points_world_homogeneous = np.hstack([points_world, np.ones((points_world.shape[0], 1))])
        points_cam_i = (pose @ points_world_homogeneous.T).T[:, :3]
        
        valid_depth_mask_proj = points_cam_i[:, 2] > 1e-5
        
        projected_pixels_homogeneous = (intrinsics @ points_cam_i[valid_depth_mask_proj].T)
        projected_pixels = projected_pixels_homogeneous[:2] / (projected_pixels_homogeneous[2] + 1e-6)
        px, py = projected_pixels[0], projected_pixels[1]

        # === 여기가 수정된 부분 ===
        mask = (px >= 0) & (px < w_img - 1) & (py >= 0) & (py < h_img - 1)
        # =======================
        
        valid_px = np.round(px[mask]).astype(int)
        valid_py = np.round(py[mask]).astype(int)
        
        colors = np.zeros((len(valid_depth_mask_proj), 3), dtype=np.float32)
        valid_indices = np.where(valid_depth_mask_proj)[0]
        colors[valid_indices[mask]] = img[valid_py, valid_px]
        all_colors[i, :] = colors

    color_diffs = []
    ref_colors = all_colors[0]
    for j in range(1, N):
        valid_mask = (np.sum(ref_colors, axis=1) > 0) & (np.sum(all_colors[j], axis=1) > 0)
        if np.any(valid_mask):
            diff = ref_colors[valid_mask] - all_colors[j][valid_mask]
            color_diffs.append(diff)

    if not color_diffs:
        return variance_map

    all_diffs = np.vstack(color_diffs)
    if len(all_diffs) < 2: return variance_map
    
    cov_matrix = np.cov(all_diffs, rowvar=False) + np.eye(3) * 1e-6
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    total_scores = np.zeros(num_valid_pixels, dtype=np.float32)
    for j in range(1, N):
        diff = ref_colors - all_colors[j]
        mahalanobis_dist_sq = np.sum((diff @ inv_cov_matrix) * diff, axis=1)
        mahalanobis_dist = np.sqrt(np.maximum(0, mahalanobis_dist_sq))
        total_scores += mahalanobis_dist
    
    final_scores = gamma / N * total_scores
    variance_map[v_coords, u_coords] = final_scores
    
    return variance_map


def main():
    parser = argparse.ArgumentParser(description="Generate and visualize a variance map from COLMAP data.")
    parser.add_argument("--colmap_path", required=True, help="Path to the COLMAP model folder (containing .bin files).")
    parser.add_argument("--image_path", required=True, help="Path to the folder containing all images.")
    parser.add_argument("--target_image_name", required=True, help="Filename of the target image.")
    # --- 저장 경로 인자 추가 ---
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output visualizations. If not provided, plots are shown but not saved.")
    args = parser.parse_args()

    # (이전과 동일한 1 ~ 6단계 로직)
    # 1. COLMAP 모델 로드 ...
    # ...
    # 6. 분산 맵 계산 ...
    print("Loading COLMAP model...")
    cameras = read_colmap_bin(os.path.join(args.colmap_path, "cameras.bin"), "cameras")
    images = read_colmap_bin(os.path.join(args.colmap_path, "images.bin"), "images")
    points3D = read_colmap_bin(os.path.join(args.colmap_path, "points3D.bin"), "points3D")
    print(f"Loaded {len(cameras)} cameras, {len(images)} images, {len(points3D)} 3D points.")

    target_info = None
    other_info_list = []
    for img_id, img_data in images.items():
        if img_data.name == args.target_image_name:
            target_info = img_data
        else:
            other_info_list.append(img_data)
    
    if not target_info:
        raise ValueError(f"Target image '{args.target_image_name}' not found in COLMAP model.")

    print("Loading images...")
    target_image = cv2.imread(os.path.join(args.image_path, target_info.name))
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB) / 255.0
    other_images = []
    for img_info in other_info_list:
        img = cv2.imread(os.path.join(args.image_path, img_info.name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        other_images.append(img)
    
    target_cam_info = cameras[target_info.camera_id]
    
    if target_cam_info.model == "SIMPLE_PINHOLE":
        fx = fy = target_cam_info.params[0]
        cx, cy = target_cam_info.params[1], target_cam_info.params[2]
    else: # PINHOLE
        fx, fy = target_cam_info.params[0], target_cam_info.params[1]
        cx, cy = target_cam_info.params[2], target_cam_info.params[3]

    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    R_ref = qvec2rotmat(target_info.qvec)
    t_ref = target_info.tvec
    ref_pose = np.eye(4)
    ref_pose[:3, :3] = R_ref
    ref_pose[:3, 3] = t_ref
    
    other_poses = []
    for img_info in other_info_list:
        R = qvec2rotmat(img_info.qvec)
        t = img_info.tvec
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        other_poses.append(pose)
    
    print("Generating sparse depth map for target image...")
    ref_depth = generate_sparse_depth_map(target_info, target_cam_info, points3D)
    
    print("Calculating variance map... (This may take a while)")
    variance_map = create_variance_map(
        target_image, ref_pose, ref_depth, other_images, other_poses, intrinsics
    )
    print("Calculation complete.")


    # 7. 시각화 및 저장
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Variance Analysis for {args.target_image_name}", fontsize=16)

    axes[0].imshow(target_image)
    axes[0].set_title('Target Image')
    axes[0].axis('off')

    axes[1].imshow(ref_depth, cmap='jet')
    axes[1].set_title('Sparse Depth Map')
    axes[1].axis('off')

    # 분산 맵의 값이 0인 부분을 검은색으로, 값이 있는 부분은 viridis 컬러맵으로 표시
    masked_variance_map = np.ma.masked_where(variance_map == 0, variance_map)
    im = axes[2].imshow(masked_variance_map, cmap='viridis')
    axes[2].set_title('Variance Map')
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- 저장 로직 ---
    if args.output_path:
        # 출력 폴더가 없으면 생성
        os.makedirs(args.output_path, exist_ok=True)
        
        # 1. 전체 시각화 창 저장
        fig_path = os.path.join(args.output_path, f"visualization_{os.path.basename(args.target_image_name)}")
        plt.savefig(fig_path)
        print(f"Saved combined visualization to: {fig_path}")

        # 2. 개별 맵 이미지 저장
        depth_path = os.path.join(args.output_path, f"depth_map_{os.path.basename(args.target_image_name)}")
        plt.imsave(depth_path, ref_depth, cmap='jet')
        print(f"Saved depth map image to: {depth_path}")
        
        variance_path = os.path.join(args.output_path, f"variance_map_{os.path.basename(args.target_image_name)}")
        plt.imsave(variance_path, masked_variance_map, cmap='viridis')
        print(f"Saved variance map image to: {variance_path}")

        # 3. (선택) 분산 맵 데이터(.npy) 저장
        variance_data_path = os.path.join(args.output_path, f"variance_map_{os.path.splitext(os.path.basename(args.target_image_name))[0]}.npy")
        np.save(variance_data_path, variance_map)
        print(f"Saved variance map data to: {variance_data_path}")

    # 화면에 표시
    plt.show()

if __name__ == '__main__':
    main()