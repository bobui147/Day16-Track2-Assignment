1. Tôi dùng Google Cloud Platform, zone `us-central1-a`, máy ảo CPU `e2-medium`, source commit `55539f67d7c78b43afe334a2ec3271c4bfdbbe2d`.
2. Dataset có 284.807 dòng; tôi chia 64% train, 16% validation và 20% test bằng stratified split với seed 42.
3. Load dữ liệu mất 3,022085 giây; training mất 8,793871 giây; best iteration là 130.
4. Trên tập test, AUC đạt 0,963446, Accuracy 0,998894, F1 0,734177, Precision 0,625899 và Recall 0,887755.
5. Latency 1 dòng là 1,566889 ms; throughput batch 1.000 dòng đạt 153.132,204 dòng/giây; model được warm-up trước, latency lấy trung vị 100 lần và throughput đo trên một batch 1.000 dòng.
6. CPU/RAM/Network tôi quan sát lúc [CẦN ĐIỀN THỜI ĐIỂM] là [CẦN ĐIỀN SỐ LIỆU TỪ `top`, `free -h`, `ip -s link`]; ảnh đính kèm [CẦN ĐIỀN TÊN ẢNH].
7. Billing tại [CẦN ĐIỀN THỜI ĐIỂM] ghi nhận [CẦN ĐIỀN DỊCH VỤ VÀ CHI PHÍ, HOẶC “CHƯA CẬP NHẬT”]; ảnh đính kèm [CẦN ĐIỀN TÊN ẢNH].
8. Tôi đã tải `benchmark_result.json` về máy; tài nguyên được xóa lúc [CHỈ ĐIỀN SAU KHI `terraform destroy` THÀNH CÔNG], với bằng chứng [CẦN ĐIỀN ẢNH `Destroy complete!`/VM KHÔNG CÒN RUNNING].
