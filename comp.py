"""
간단한 FunASR SenseVoice 모델 압축 스크립트
"""

import os
import torch
import torch.quantization as quant
import argparse
from pathlib import Path
from funasr import AutoModel

def load_finetuned_model(model_path, device="cpu"):
    """파인튜닝된 모델 로드"""
    print(f"Loading model: {model_path}")
    
    try:
        # 원본 모델 구조 로드
        model = AutoModel(model="iic/SenseVoiceSmall", device=device, disable_update=True)
        
        # 파인튜닝된 가중치 로드
        checkpoint = torch.load(model_path, map_location="cpu")
        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            weights = checkpoint['model']
        else:
            weights = checkpoint
        
        model.model.load_state_dict(weights, strict=False)
        print("✓ Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def compress_fp16(model, output_path):
    """Float16 압축"""
    print("Compressing to Float16...")
    model_fp16 = model.half()
    torch.save(model_fp16.model.state_dict(), output_path)
    print(f"✓ Saved: {output_path}")
    return model_fp16

def compress_int8(model, output_path, selective=True):
    """INT8 압축"""
    mode = "selective" if selective else "full"
    print(f"Compressing to INT8 ({mode})...")
    
    layers = {torch.nn.Linear} if selective else {torch.nn.Linear, torch.nn.Conv1d}
    quantized = quant.quantize_dynamic(model.cpu().model, layers, dtype=torch.qint8)
    
    torch.save(quantized, output_path)
    print(f"✓ Saved: {output_path}")
    return quantized

def main():
    parser = argparse.ArgumentParser(description="Compress FunASR model")
    parser.add_argument("--model_path", default="./outputs/model.pt", help="Model path")
    parser.add_argument("--method", default="all", choices=["fp16", "int8", "all", "benchmark"], help="Compression method")
    parser.add_argument("--output_dir", default="./compressed", help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device")
    
    args = parser.parse_args()
    
    # 모델 로드
    model = load_finetuned_model(args.model_path, args.device)
    if model is None:
        return
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_name = Path(args.model_path).stem
    
    print(f"\nModel: {args.model_path}")
    print(f"Method: {args.method}")
    print("-" * 40)
    
    # 압축 실행
    if args.method in ["fp16", "all", "benchmark"]:
        compress_fp16(model, output_dir / f"{model_name}_fp16.pt")
    
    if args.method in ["int8", "all", "benchmark"]:
        compress_int8(model, output_dir / f"{model_name}_int8_selective.pt", selective=True)
        compress_int8(model, output_dir / f"{model_name}_int8_full.pt", selective=False)
    
    print("\n✓ Compression completed!")
    print(f"Check: {output_dir}")

if __name__ == "__main__":
    main()