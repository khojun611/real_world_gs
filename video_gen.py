from PIL import Image
import os

def images_to_gif(image_folder, output_gif, fps=30):
    """
    이미지 폴더에서 이미지를 읽어 GIF 파일을 생성합니다.
    손상된 파일은 자동으로 건너뜁니다.
    """
    # 이미지 파일 목록을 불러와 이름순으로 정렬
    try:
        images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg"))]
        images.sort()
    except FileNotFoundError:
        print(f"오류: '{image_folder}' 폴더를 찾을 수 없습니다.")
        return

    # 이미지 프레임 불러오기
    frames = []
    for image_name in images:
        img_path = os.path.join(image_folder, image_name)
        try:
            frame = Image.open(img_path)
            frame.load()  # 이미지 데이터 전체를 불러와 파일 무결성 검사
            frames.append(frame)
        except OSError as e:
            print(f"손상된 파일을 건너뜁니다: {img_path} - {e}")
            continue  # 다음 파일로 이동

    # 프레임이 없으면 함수 종료
    if not frames:
        print("GIF를 만들 유효한 이미지가 없습니다.")
        return
        
    # GIF 파일로 저장
    duration = int(1000 / fps)  # 프레임당 지속 시간 (밀리초)
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0  # 무한 반복
    )
    print(f"GIF 파일이 {output_gif} 이름으로 저장되었습니다.")

# --- 사용자 설정 ---
# 이미지 폴더 경로와 출력 GIF 파일 경로 지정
image_folder = '/home/iris/ref-gaussian/output_abc/gardenspheres/gardenspheres-0711_0448/all_renders/grid'
output_gif = './garden_refgau.gif'

# 함수 호출 (초당 프레임(FPS)은 30으로 설정)
images_to_gif(image_folder, output_gif, fps=30)