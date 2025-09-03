#!/usr/bin/env python3

import sys
import os
sys.path.append('./')

import torch
from funasr import AutoModel
from funasr.datasets.sense_voice_datasets.datasets import SenseVoiceDataset
from funasr.frontends.wav_frontend import WavFrontend
from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer

import os
import csv
import glob
import librosa
import soundfile as sf


def evaluate_models():
    """Base 모델과 학습된 모델의 정확도를 평가 (train_data_audio_sample_index.csv 기반)"""
    try:
        import jiwer
    except ImportError:
        print("[ERROR] jiwer 패키지가 설치되어 있지 않습니다. 'pip install jiwer'로 설치 후 실행하세요.")
        return
    print("\n=== Evaluating Base Model vs. Fine-tuned Model ===")
    csv_data = "./train_data_audio_sample_index.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # base 모델 준비
    base_model = AutoModel(model="iic/SenseVoiceSmall", disable_update=True)
    tokenizer = SentencepiecesTokenizer(
        bpemodel=base_model.kwargs.get('tokenizer_conf', {}).get('bpemodel', ''),
        unk_symbol="<unk>",
        split_with_space=True
    )

    valid_samples = []
    with open(csv_data, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row.get('Index') or row.get('\ufeffIndex')
            text = row.get('Query(한글)')
            if idx and text:
                wav_candidates = glob.glob(os.path.join("./audio", f"{idx}*.wav"))
                wav_path = None
                # 우선순위: level2 > noise > 나머지
                for pattern in [f"{idx}_level2.wav", f"{idx}_noise.wav"]:
                    candidate = os.path.join("./audio", pattern)
                    if candidate in wav_candidates:
                        wav_path = candidate
                        break
                if not wav_path and wav_candidates:
                    wav_path = wav_candidates[0]
                if wav_path and os.path.exists(wav_path):
                    valid_samples.append({'source': wav_path, 'target': text})
                    print(f"index: {idx}, text: {text}")
                    print(f"wav_path: {wav_path}")
                else:
                    print(f"[WARNING] No audio file found for index: {idx}")

    # 학습된 모델 준비 (최적 모델 사용)
    finetuned_model = AutoModel(model="iic/SenseVoiceSmall", disable_update=True)
    finetuned_dirs = glob.glob("../outputs")
    if finetuned_dirs:
        latest_dir = max(finetuned_dirs)
        print(f"Found latest finetuned directory: {latest_dir}")
        model_paths = [
            f"{latest_dir}/model..pt",
        ]
        checkpoint_files = glob.glob(f"{latest_dir}/model.pt.ep*")
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
            model_paths.extend(checkpoint_files)
    else:
        print("No finetuned directories found in ./outputs/, falling back to old path")
        model_paths = [
            "../outputs/model.pt",
            "../outputs/model.pt.best"
        ]
    loaded_model = None
    for finetuned_path in model_paths:
        if os.path.exists(finetuned_path):
            try:
                print(f"Loading model from: {finetuned_path}")
                state = torch.load(finetuned_path, map_location=device)
                finetuned_model.model.load_state_dict(state['model_state_dict'])
                loaded_model = finetuned_path
                if 'training_stats' in state:
                    stats = state['training_stats']
                    print(f"  - Steps: {stats.get('step', 'N/A')}")
                    print(f"  - Samples: {stats.get('total_samples', 'N/A')}")
                    if 'losses' in stats and len(stats['losses']) > 0:
                        print(f"  - Final loss: {stats['losses'][-1]:.4f}")
                break
            except Exception as e:
                print(f"Failed to load {finetuned_path}: {e}")
                continue
    if loaded_model is None:
        print("[ERROR] 사용 가능한 학습된 모델이 없습니다!")
        return
    print(f"Successfully loaded: {loaded_model}")

    base_model.model.to(device)
    finetuned_model.model.to(device)
    base_model.model.eval()
    finetuned_model.model.eval()

    import re
    def extract_pure_text(val):
        if isinstance(val, list):
            if val and isinstance(val[0], dict) and 'text' in val[0]:
                val = val[0]['text']
            else:
                val = str(val)
        if isinstance(val, dict):
            if 'text' in val:
                val = val['text']
            elif 'result' in val:
                val = val['result']
            else:
                val = str(val)
        if isinstance(val, str):
            val = re.sub(r'<\|.*?\|>', '', val)
            val = val.strip()
        return val

    def infer_text(model, wav_path):
        try:
            # 16kHz로 리샘플링하여 임시 파일로 저장
            y, sr = librosa.load(wav_path, sr=16000)
            tmp_wav = wav_path + ".tmp16k.wav"
            sf.write(tmp_wav, y, 16000)
            result = model.generate(input=tmp_wav, cache={}, language="ko")
            os.remove(tmp_wav)
            return extract_pure_text(result)
        except Exception as e:
            print(f"[ERROR] {wav_path} 추론 실패: {e}")
            return ""

    # 전체 샘플 평가 및 csv 저장
    csv_rows = []
    for i, sample in enumerate(valid_samples):
        ref = sample['target']
        wav_path = sample['source']
        if wav_path and not os.path.isabs(wav_path):
            wav_path = os.path.join(os.path.dirname(csv_data), wav_path)
        base_hyp = infer_text(base_model, wav_path) if wav_path and os.path.exists(wav_path) else ""
        fine_hyp = infer_text(finetuned_model, wav_path) if wav_path and os.path.exists(wav_path) else ""
        ref_pure = extract_pure_text(ref)
        csv_rows.append({
            'wav': os.path.basename(wav_path) if wav_path else '',
            'base_result': base_hyp,
            'fine_result': fine_hyp,
            'ref': ref_pure
        })
    csv_path = "eval_result.csv"
    with open(csv_path, "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ['wav', 'base_result', 'fine_result', 'ref']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    # WER, CER 계산
    base_refs, base_hyps = [], []
    fine_refs, fine_hyps = [], []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = row['ref']
            base_hyp = row['base_result']
            fine_hyp = row['fine_result']
            base_refs.append(ref)
            base_hyps.append(base_hyp)
            fine_refs.append(ref)
            fine_hyps.append(fine_hyp)
    base_wer = jiwer.wer(base_refs, base_hyps)
    fine_wer = jiwer.wer(fine_refs, fine_hyps)
    base_cer = jiwer.cer(base_refs, base_hyps)
    fine_cer = jiwer.cer(fine_refs, fine_hyps)

    print(f"base_wer: {base_wer:.4f}")
    print(f"base_cer: {base_cer:.4f}")
    print(f"finetuned_wer: {fine_wer:.4f}")
    print(f"finetuned_cer: {fine_cer:.4f}")

    import shutil
    import re

    # eval_result.csv 복제
    csv_path_nospace = "eval_result_nospace.csv"
    shutil.copy("eval_result.csv", csv_path_nospace)

    # 공백 및 마침표, 물음표 제거 함수
    def clean_text(s):
        if not isinstance(s, str):
            return s
        return re.sub(r"[ .?]", "", s)

    base_refs_ns, base_hyps_ns = [], []
    fine_refs_ns, fine_hyps_ns = [], []
    rows = []

    with open(csv_path_nospace, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['base_result'] = clean_text(row['base_result'])
            row['fine_result'] = clean_text(row['fine_result'])
            row['ref'] = clean_text(row['ref'])
            base_refs_ns.append(row['ref'])
            base_hyps_ns.append(row['base_result'])
            fine_refs_ns.append(row['ref'])
            fine_hyps_ns.append(row['fine_result'])
            rows.append(row)

    # 덮어쓰기(공백 및 . ? 제거된 값으로)
    with open(csv_path_nospace, "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ['wav', 'base_result', 'fine_result', 'ref']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # 공백 및 . ? 제거 후 WER/CER
    base_wer_ns = jiwer.wer(base_refs_ns, base_hyps_ns)
    fine_wer_ns = jiwer.wer(fine_refs_ns, fine_hyps_ns)
    base_cer_ns = jiwer.cer(base_refs_ns, base_hyps_ns)
    fine_cer_ns = jiwer.cer(fine_refs_ns, fine_hyps_ns)

    print("\n[공백 및 . ? 제거 후 평가 결과]")
    print(f"base_wer_nospace: {base_wer_ns:.4f}")
    print(f"base_cer_nospace: {base_cer_ns:.4f}")
    print(f"finetuned_wer_nospace: {fine_wer_ns:.4f}")
    print(f"finetuned_cer_nospace: {fine_cer_ns:.4f}")

if __name__ == "__main__":
    evaluate_models()
