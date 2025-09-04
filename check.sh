#!/bin/bash

echo "🔍 SenseVoice 데이터 문제 진단"
echo "================================"

# 1. 파일 존재 확인
echo "1. 파일 존재 확인:"
# data dir, which contains: train.json, val.json
train_data=${workspace}/train_data/data/train_data.jsonl
val_data=${workspace}/train_data/data/valid_data.jsonl


if [ -f "$train_data" ]; then
    echo "✅ Train 파일 존재: $(wc -l < $train_data) 줄"
else
    echo "❌ Train 파일 없음: $train_data"
fi

if [ -f "$val_data" ]; then
    echo "✅ Val 파일 존재: $(wc -l < $val_data) 줄"
else
    echo "❌ Val 파일 없음: $val_data"
fi

# 2. 파일 크기 확인
echo ""
echo "2. 파일 크기 확인:"
du -sh $train_data $val_data 2>/dev/null || echo "파일 크기 확인 실패"

# 3. 데이터 샘플 확인
echo ""
echo "3. Train 데이터 샘플:"
head -2 $train_data 2>/dev/null || echo "Train 파일 읽기 실패"

echo ""
echo "4. Val 데이터 샘플:"
head -2 $val_data 2>/dev/null || echo "Val 파일 읽기 실패"

# 5. 대안 경로 찾기
echo ""
echo "5. JSONL 파일 찾기:"
find /giant-data/user/1112307/finetune/SenseVoice -name "*.jsonl" -type f 2>/dev/null | head -10

echo ""
echo "================================"