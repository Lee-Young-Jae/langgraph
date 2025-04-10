import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
from transformers import AutoModelForImageSegmentation

# utils.py에서 필요한 함수
def check_state_dict(state_dict):
    # 모델의 state_dict를 확인하고 필요한 경우 수정
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # 'module.' 제거
        new_state_dict[k] = v
    return new_state_dict

def convert_weights(state_dict):
    # 현재 모델 구조에 맞게 가중치 변환
    converted_dict = {}
    # backbone 가중치를 conv1에 매핑
    if 'backbone.conv1.weight' in state_dict:
        converted_dict['conv1.weight'] = state_dict['backbone.conv1.weight']
        converted_dict['conv1.bias'] = state_dict['backbone.conv1.bias']
    # 나머지 가중치도 유사하게 매핑
    if 'backbone.conv2.weight' in state_dict:
        converted_dict['conv2.weight'] = state_dict['backbone.conv2.weight']
        converted_dict['conv2.bias'] = state_dict['backbone.conv2.bias']
    if 'backbone.conv3.weight' in state_dict:
        converted_dict['conv3.weight'] = state_dict['backbone.conv3.weight']
        converted_dict['conv3.bias'] = state_dict['backbone.conv3.bias']
    return converted_dict

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')

# 모델 로드
model = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
model.eval()
model.to(device)
model.half()
print('BiRefNet is ready to use.')

# 이미지 전처리
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 이미지 처리 함수
def process_image(input_path, output_path):
    print("이미지 로드 시작...")
    try:
        # 이미지 로드 및 전처리
        image = Image.open(input_path)
        print(f"이미지 크기: {image.size}")
        image = image.convert("RGB") if image.mode != "RGB" else image
        
        input_tensor = transform_image(image).unsqueeze(0).to(device)
        input_tensor = input_tensor.half()
        print("모델 추론 시작...")

        # 추론
        with torch.no_grad():
            preds = model(input_tensor)[-1].sigmoid().cpu()
        print("추론 완료")
        
        pred = preds[0].squeeze()

        # 결과 저장
        print("결과 저장 중...")
        pred_pil = transforms.ToPILImage()(pred)
        pred_pil = pred_pil.resize(image.size)
        pred_pil.save(output_path)
        print("저장 완료")

        return pred_pil
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        raise

# 사용 예시
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_image_path = os.path.join(current_dir, 'input.jpg')
    output_image_path = os.path.join(current_dir, 'output.png')
    print(f"입력 이미지 경로: {input_image_path}")
    print(f"출력 이미지 경로: {output_image_path}")
    process_image(input_image_path, output_image_path)
    print("이미지 처리 완료")