# Public Paper Ai 2024

Theo thống kê trên trang paperswithcode (https://paperswithcode.com/) hiện có 16 tasks lớn mà Machine Learning có thể thực
hiện trong đó có tới trên 8 tasks Deep learning đạt kết quả SOTA phải kể đến như:
• Computer Vision
• Natural Language Processing
• Medical
• Methodology
• Speech
• Time Series
• Audio
• Music


#### -------- https://paperswithcode.com/ -----
 - Danh sách Top các bài báo + Code github nghiên cứu ML / AI mới nhất
 - Cung Cấp Code , Paper , Dataset ==> Sẵn sàng làm và nghiên cứu

#### ---------- Danh sách TOPIC NGhiên cứu ứng dụng Công nghệ mới AI --------------

## 1.TOPIC 1: 
  RAG một thành phần truy xuất thông tin .
  được giới thiệu sử dụng đầu vào của người dùng để lấy trước thông tin từ một nguồn dữ liệu mới.
  Tham Khảo : aws : https://aws.amazon.com/vi/what-is/retrieval-augmented-generation/
                    https://aws.amazon.com/vi/what-is/large-language-model/
              aks : code trên github
			  Google : code trên github
  
   ### V1.0 : Advanded RAG : 

   # Viết Code :--> V2.0 Auto fixed/ Future Code
   https://paperswithcode.com/paper/qwen2-5-coder-technical-report

## 2. TOPIC 2 LOW-CODE : sử dụng Curso AI / Copilot / Google Assistan Code / qwen2-5-coder-technical-report
   - V1.0 : Hỗ trợ Dev fixed / viết tính năng ,... : Tối ưu code , services hiện có
           + Tham khảo trên 80.000 line code 
		   + và làm / sửa trực tiếp trên 10- 100 Micro services (EbankX và B2B) TP Bank đang chạy
   - Vx.x : + Auto fixed / Write Code ==> Công ty phần mền ảo
            + Tham khảo : ChatDev / SuperAGI / CrewAI / Dewin / .... 

# Trợ lý Text2Speed , Speed2Text
https://paperswithcode.com/paper/mini-omni-language-models-can-hear-talk-while
https://github.com/gpt-omni/mini-omni

Mini-Omni là một mô hình ngôn ngữ lớn đa phương thức nguồn mở có thể nghe, nói trong khi suy nghĩ . 
Có khả năng nhập giọng nói đầu cuối theo thời gian thực và khả năng đàm thoại đầu ra âm thanh phát trực tuyến .

Mini-Omni , mô hình ngôn ngữ lớn đa mô hình nguồn mở đầu tiên có khả năng đàm thoại thời gian thực, 
   có khả năng nhập và xuất giọng nói hoàn toàn từ đầu đến cuối. 
   Nó cũng bao gồm nhiều chức năng chuyển âm thanh thành văn bản khác như Nhận dạng giọng nói tự động (ASR)

#### Bắt đầu nhanh  : khởi động máy chủ
LƯU Ý: bạn cần khởi động máy chủ trước khi chạy bản demo streamlit hoặc gradio với API_URL được đặt thành địa chỉ máy chủ.

```cmd
sudo apt-get install ffmpeg
conda activate omni
cd mini-omni
python3 server.py --ip '0.0.0.0' --port 60808
```

### chạy bản demo streamlit
LƯU Ý: bạn cần chạy streamlit cục bộ với PyAudio đã cài đặt. Đối với lỗi: ModuleNotFoundError: No module named 'utils.vad', vui lòng chạy export PYTHONPATH=./trước.

pip install PyAudio==0.2.14
API_URL=http://0.0.0.0:60808/chat streamlit run webui/omni_streamlit.py

### chạy bản demo gradio
API_URL=http://0.0.0.0:60808/chat python3 webui/omni_gradio.py


## Fine Tunring / Finetune : Tinh chỉnh
### --------Lightning-AI/litgpt-----------
https://github.com/Lightning-AI/litgpt/
Tinh chỉnh là quá trình lấy một mô hình AI đã được đào tạo trước 
và đào tạo thêm trên một tập dữ liệu nhỏ hơn, chuyên biệt hơn, phù hợp với một tác vụ 
hoặc ứng dụng cụ thể.

# 0) setup your dataset
curl -L https://huggingface.co/datasets/ksaw008/finance_alpaca/resolve/main/finance_alpaca.json -o my_custom_dataset.json

# 1) Finetune a model (auto downloads weights)
litgpt finetune microsoft/phi-2 \
  --data JSON \
  --data.json_path my_custom_dataset.json \
  --data.val_split_fraction 0.1 \
  --out_dir out/custom-model

# 2) Test the model
litgpt chat out/custom-model/final

# 3) Deploy the model
litgpt serve out/custom-model/final