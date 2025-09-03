import librosa
import soundfile as sf
import json
import os
import numpy as np
from pathlib import Path
import pandas as pd

def augment_single_audio(audio_path, text, output_dir, audio_id):
    """
    하나의 오디오 파일을 여러 방법으로 증강
    
    Args:
        audio_path: 원본 오디오 경로
        text: 전사 텍스트  
        output_dir: 증강 파일 저장 디렉터리
        audio_id: 고유 ID
        
    Returns:
        증강된 데이터 리스트
    """
    
    print(f"처리 중: {audio_path}")
    
    try:
        # 1. 원본 오디오 로드
        y, sr = librosa.load(audio_path, sr=16000)
        print(f"  원본: {len(y)} 샘플, {len(y)/sr:.2f}초")
        
        augmented_data = []
        
        # 2. 원본도 포함 (포맷 통일)
        original_output = f"{output_dir}/{audio_id}_original.wav"
        sf.write(original_output, y, sr)
        augmented_data.append({
            "key": f"{audio_id}_original",
            "source": f"{audio_id}_original.wav",  # 상대 경로로 저장
            "target": text,
            "text_language": "<|ko|>"
        })
        
        # 3. 속도 변경 증강
        speeds = [0.8, 0.9, 1.1, 1.2]  # 원본(1.0) 제외
        for speed in speeds:
            print(f"  속도 {speed}배 처리 중...")
            
            try:
                # scipy를 이용한 리샘플링으로 속도 변경 구현
                from scipy.signal import resample
                # 속도가 빠르면 더 적은 샘플, 느리면 더 많은 샘플
                new_length = int(len(y) / speed)
                y_speed = resample(y, new_length)
                # 원래 길이와 다르므로 실제 변경된 배율 계산
                actual_rate = len(y) / len(y_speed)
                print(f"    실제 속도: {actual_rate:.2f}배")
            except Exception as e:
                print(f"    속도 변경 실패: {e}, 원본 사용")
                y_speed = y
            
            # 새 파일로 저장
            speed_output = f"{output_dir}/{audio_id}_speed_{speed}.wav"
            sf.write(speed_output, y_speed, sr)
            
            print(f"    → {len(y_speed)} 샘플, {len(y_speed)/sr:.2f}초로 변경")
            
            # 메타데이터 추가
            augmented_data.append({
                "key": f"{audio_id}_speed_{speed}",
                "source": f"{audio_id}_speed_{speed}.wav",  # 상대 경로로 저장
                "target": text,  # 텍스트는 동일
                "text_language": "<|ko|>"
            })
        
        # 4. 볼륨 변경 증강 (피치 대신 더 간단하고 효과적)
        volumes = [0.7, 1.3]  # 70%, 130% 볼륨
        for i, volume in enumerate(volumes):
            print(f"  볼륨 {volume:.1f}배 처리 중...")
            
            try:
                # 볼륨 조정
                y_volume = y * volume
                # 클리핑 방지
                y_volume = np.clip(y_volume, -1.0, 1.0)
                print(f"    볼륨 조정 완료")
            except Exception as e:
                print(f"    볼륨 변경 실패: {e}, 원본 사용")
                y_volume = y
            
            # 새 파일로 저장
            volume_output = f"{output_dir}/{audio_id}_volume_{i}.wav"
            sf.write(volume_output, y_volume, sr)
            
            # 메타데이터 추가
            augmented_data.append({
                "key": f"{audio_id}_volume_{i}",
                "source": f"{audio_id}_volume_{i}.wav",  # 상대 경로로 저장
                "target": text,  # 텍스트는 동일
                "text_language": "<|ko|>"
            })
        
        # 5. 노이즈 추가 증강
        noise_levels = [0.005, 0.01, 0.02]  # SNR ~20dB, ~10dB
        for i, noise_level in enumerate(noise_levels):
            print(f"  노이즈 레벨 {noise_level} 처리 중...")
            
            # 가우시안 노이즈 생성 및 추가
            noise = np.random.normal(0, noise_level, len(y))
            y_noise = y + noise
            
            # 클리핑 방지
            y_noise = np.clip(y_noise, -1.0, 1.0)
            
            # 새 파일로 저장
            noise_output = f"{output_dir}/{audio_id}_noise_{i}.wav"
            sf.write(noise_output, y_noise, sr)
            
            # 메타데이터 추가
            augmented_data.append({
                "key": f"{audio_id}_noise_{i}",
                "source": f"{audio_id}_noise_{i}.wav",  # 상대 경로로 저장
                "target": text,  # 텍스트는 동일
                "text_language": "<|ko|>"
            })
        
        print(f"  완료: {len(augmented_data)}개 파일 생성")
        return augmented_data
        
    except Exception as e:
        print(f"  오류 발생: {e}")
        return []

def process_entire_dataset(input_dir, output_dir, train_ratio=0.9):
    """
    전체 데이터셋을 증강하고 train/valid JSONL 생성
    """
    
    # 출력 디렉터리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # train/valid JSONL 파일 경로
    train_input = os.path.join(input_dir, 'train_data.jsonl')
    valid_input = os.path.join(input_dir, 'valid_data.jsonl')
    
    all_augmented = []
    
    # 1. 훈련 데이터 증강
    if os.path.exists(train_input):
        print(f"훈련 데이터 처리: {train_input}")
        with open(train_input, 'r', encoding='utf-8') as f:
            train_data = [json.loads(line) for line in f]
        
        print(f"원본 훈련 데이터: {len(train_data)}개")
        
        for i, data in enumerate(train_data):
            print(f"\n[훈련 {i+1}/{len(train_data)}] 처리 중...")
            
            audio_id = data['key']
            # 입력 디렉터리 기준으로 오디오 경로 구성
            audio_path = os.path.join(input_dir, data['source'])
            text = data['target']
            
            # 오디오 파일 존재 확인
            if not os.path.exists(audio_path):
                print(f"  파일 없음: {audio_path}")
                continue
            
            # 단일 오디오 증강
            augmented = augment_single_audio(audio_path, text, output_dir, audio_id)
            all_augmented.extend(augmented)
    
    # 2. 검증 데이터 증강
    if os.path.exists(valid_input):
        print(f"\n검증 데이터 처리: {valid_input}")
        with open(valid_input, 'r', encoding='utf-8') as f:
            valid_data = [json.loads(line) for line in f]
        
        print(f"원본 검증 데이터: {len(valid_data)}개")
        
        for i, data in enumerate(valid_data):
            print(f"\n[검증 {i+1}/{len(valid_data)}] 처리 중...")
            
            audio_id = data['key']
            # 입력 디렉터리 기준으로 오디오 경로 구성
            audio_path = os.path.join(input_dir, data['source'])
            text = data['target']
            
            # 오디오 파일 존재 확인
            if not os.path.exists(audio_path):
                print(f"  파일 없음: {audio_path}")
                continue
            
            # 단일 오디오 증강
            augmented = augment_single_audio(audio_path, text, output_dir, audio_id)
            all_augmented.extend(augmented)
    
    # 3. 증강된 데이터를 train/valid로 다시 분할
    import random
    random.shuffle(all_augmented)
    
    total_samples = len(all_augmented)
    train_size = int(total_samples * train_ratio)
    
    train_augmented = all_augmented[:train_size]
    valid_augmented = all_augmented[train_size:]
    
    # 4. 새로운 JSONL 파일들 저장
    train_output = os.path.join(output_dir, 'train_data.jsonl')
    valid_output = os.path.join(output_dir, 'valid_data.jsonl')
    
    with open(train_output, 'w', encoding='utf-8') as f:
        for data in train_augmented:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    with open(valid_output, 'w', encoding='utf-8') as f:
        for data in valid_augmented:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"\n=== 증강 완료 ===")
    print(f"총 증강된 샘플: {total_samples}개")
    print(f"훈련 데이터: {len(train_augmented)}개 → {train_output}")
    print(f"검증 데이터: {len(valid_augmented)}개 → {valid_output}")
    
    try:
        if 'train_data' in locals() and 'valid_data' in locals():
            original_total = len(train_data) + len(valid_data)
            print(f"증강 비율: {total_samples/original_total:.1f}배")
    except:
        pass

# 실행
if __name__ == "__main__":
    process_entire_dataset(
        input_dir='./data',            # 입력: ./data/ 폴더 (train_data.jsonl, valid_data.jsonl, *.wav)
        output_dir='./data-augmented'  # 출력: ./data-augmented/ 폴더
    )