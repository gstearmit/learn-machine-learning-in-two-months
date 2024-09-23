
# repo : https://github.com/gstearmit/traffic-sign-recognition/blob/master/README_VI.md

# Để triển khai thực tế dự án thì cần data cho vào 3 thư mục : train, test, validation 
tham khảo video: https://youtu.be/ee9tF9xEf04?list=PLMm4sOMuA2QI5x_0KlNT3LuKDi6-ByboB để
 "Tự động đánh nhãn và xây dựng nhanh model nhận diện ảnh trong thực tế "

# Nhận diện Biển Báo Giao Thông

## Mô Tả Dự Án & Mục Tiêu
Nhận diện biển báo giao thông có thể giúp lái xe theo nhiều cách để tăng cường nhận thức về tình trạng đường hiện tại và cải thiện an toàn bằng cách cảnh báo họ đảm bảo rằng các quy tắc không bị vi phạm. Ví dụ, nhận diện giới hạn tốc độ, biển cấm vào hoặc biển nhường đường chỉ là một vài biển báo có thể rất quan trọng để giữ an toàn cho giao thông. Một camera hướng về phía trước, quét đường phía trước, nhận diện bất kỳ biển báo nào và có thể thực hiện các hành động từ cảnh báo trên bảng điều khiển, đến việc kiểm soát tốc độ hành trình bằng cách tự động giảm tốc để đảm bảo tốc độ hiện tại không vượt quá giới hạn. Để làm cho việc nhận diện biển báo trở nên khả thi, một số phương pháp học máy được áp dụng để đào tạo một mô hình có thể thực hiện việc này.

![](https://imgur.com/Ov3v35E.gif)

## Chạy Dự Án
Dự án này được phát triển bằng Google Colab. Colab cung cấp một môi trường linh hoạt để phát triển các dự án mà được đồng bộ hóa với tài khoản Google của người dùng và được tải lên Google Drive một cách an toàn. Tuy nhiên, có lẽ điều tốt nhất là nó cho phép người dùng tận dụng GPU của Google để tăng cường hiệu suất khi chạy một dự án. Hãy chắc chắn bật tăng tốc phần cứng GPU bằng cách điều hướng đến "Runtime" -> "Change runtime type" và chọn "GPU" trong menu Tăng tốc phần cứng. Điều này có thể tiết kiệm một lượng thời gian đáng kể, đặc biệt là khi nói đến học máy. Đó là lý do tại sao, dự án này chỉ bao gồm một notebook Python duy nhất [trafficSigns.ipynb](trafficSigns.ipynb). Sau khi sao chép kho lưu trữ này bằng cách chạy
```
git clone https://github.com/canozcivelek/traffic-sign-recognition.git
```
trên dòng lệnh, dự án gần như đã sẵn sàng để chạy. Nếu muốn chạy dự án này trên Google Colab, hãy chắc chắn rằng bạn có một tài khoản Google. Sau đó, chỉ cần điều hướng đến https://colab.research.google.com, tải lên dự án và chạy tất cả các ô. Nhiều ghi chú về cách mã hoạt động được cung cấp trong phần tiếp theo.

## Cách Mã Hoạt Động
### Thu Thập Dữ Liệu
Sau khi khởi động notebook trên Colab hoặc Jupyter, ô đầu tiên sao chép các tài nguyên cần thiết của tất cả các biển báo giao thông và nhãn của chúng từ một kho lưu trữ Github. Điều này được theo sau bởi việc nhập các thư viện như numpy, matplotlib và keras. Tiếp theo, dữ liệu được tải để có thể làm việc và phân tích chúng.

### Phân Tích Dữ Liệu
Khi kiểm tra, có thể hiểu rằng tất cả các hình ảnh biển báo giao thông có kích thước 32x32 và 3 kênh màu (RGB). Tuy nhiên, để có thể đào tạo mô hình, cần phải tiền xử lý các hình ảnh theo cách mà dễ dàng hơn cho phần đào tạo và không tốn quá nhiều thời gian. Tệp [signnames.csv](signnames.csv) chứa tất cả 43 biển báo và nhãn tương ứng của chúng. Ví dụ, biển báo "Giới hạn Tốc độ 50km/h" có số "2" là nhãn của nó, hoặc biển báo "Cấm Vào" có nhãn "17". Bằng cách vẽ số lượng mẫu, có thể thấy lượng dữ liệu được cung cấp cho mỗi biển báo giao thông. Số mẫu dao động từ khoảng 250-2000 cho mỗi biển báo giao thông. Trong khi 2000 mẫu có vẻ là một lượng dữ liệu hợp lý, một số 200 mẫu rõ ràng là ít hơn cần thiết để đạt được kết quả tốt. Đó là lý do tại sao, trong các bước tiếp theo, việc tăng cường dữ liệu sẽ diễn ra để tạo ra nhiều biến thể hơn của các mẫu hiện có.

![alt text](https://github.com/canozcivelek/traffic-sign-recognition/blob/master/Images/trainDataset.png)

_Graph cho thấy phân phối của tập dữ liệu huấn luyện._

### Tiền Xử Lý Hình Ảnh
Như đã đề cập trước đó, một quá trình tiền xử lý là cần thiết để chuẩn bị tập dữ liệu cho việc đào tạo. 
* Đầu tiên, chuyển đổi chúng thành hình ảnh grayscale sẽ loại bỏ bố cục 3 kênh thành bố cục kênh đơn mong muốn. 
* Áp dụng cân bằng histogram sẽ phân phối đều mật độ mỗi pixel và chuẩn hóa ánh sáng, điều này sẽ đảm bảo rằng các hình ảnh trông ít lộn xộn và có tổ chức hơn.
* Chuẩn hóa mật độ mỗi pixel bằng cách chia chúng cho 255. Thông thường, mật độ dao động từ 0-255, sau khi thực hiện phép chia, tất cả các mật độ sẽ dao động giữa 0-1.
Các bước này được thực hiện dưới hàm preprocess() và chuẩn bị các hình ảnh cho việc đào tạo.

![alt text](https://github.com/canozcivelek/traffic-sign-recognition/blob/master/Images/imageSamples.jpg)

_Hình ảnh thô (bên trái), và hình ảnh đã được tiền xử lý (bên phải)._

### Tăng Cường Dữ Liệu
Có thể tăng cường tập dữ liệu bằng cách thực hiện một vài thay đổi trên mỗi hình ảnh. Điều này sẽ giúp tạo ra nhiều hình ảnh hơn để đào tạo, do đó, đạt được tỷ lệ học chính xác hơn. Sử dụng thư viện **keras.preprocessing.image**, **ImageDataGenerator** được nhập và thư viện này sẽ cho phép tăng cường dữ liệu bằng cách dịch hình ảnh, phóng to/thu nhỏ, cắt xén và xoay chúng ngẫu nhiên.

### Triển Khai Mô Hình _LeNet_
Mô hình LeNet được sử dụng để thực hiện việc đào tạo. Mô hình này đã được chứng minh là một mô hình hiệu quả và cung cấp tỷ lệ chính xác cao. Dưới đây là hình ảnh minh họa mô hình LeNet.

![alt text](https://github.com/canozcivelek/traffic-sign-recognition/blob/master/Images/leNet.jpg)
---

**Thêm Các Lớp Tích Chập**

Trong mô hình _sequential_ này, trước tiên thêm 2 lớp tích chập 2D như dòng _model.add(Conv2D(60, (5, 5), input_shape = (32, 32, 1), activation = 'relu'))_, mỗi lớp bao gồm 60 bộ lọc có trách nhiệm trích xuất các đặc trưng từ các hình ảnh huấn luyện. Những đặc trưng này đóng vai trò thiết yếu trong việc **dự đoán** hình dạng của một biển báo và phân loại chúng chính xác. Chúng có kích thước 5x5 sẽ quét qua mỗi hình ảnh, được giảm xuống còn 28x28 về kích thước. Tham số tiếp theo xác định hình dạng hình ảnh đầu vào được định nghĩa trước đó là 32x32x1. Cuối cùng, các lớp được kích hoạt bằng hàm kích hoạt "relu". 

**Thêm Một Lớp Pooling**

Theo kiến trúc mô hình LeNet, một lớp pooling được thêm vào để giảm kích thước bản đồ đặc trưng từ 5x5 xuống 2x2. Về cơ bản, điều này sẽ ngăn ngừa hiện tượng overfitting bằng cách có các phiên bản tổng quát hơn của các đặc trưng đã được trích xuất trước đó và cung cấp ít tham số hơn để làm việc.

**Các Lớp Tích Chập Khác**

Thêm 2 lớp tích chập 2D nữa, lần này, có 30 bộ lọc mỗi lớp với kích thước 3x3. Chúng lại được theo sau bởi một lớp pooling với kích thước pooling là 2x2.

**Các Lớp Dropout**

Một lớp dropout được thêm vào để làm cho một số nút đầu vào bị loại bỏ. Tỷ lệ dropout 0.5 có nghĩa là tại mỗi lần cập nhật, một nửa các nút đầu vào sẽ bị loại bỏ. Điều này sẽ tăng tốc quá trình và không có tác động lớn đến việc học.

**Lớp Flatten & Dense**

Bằng cách thêm một lớp flatten, dữ liệu được định dạng đúng để được đưa vào lớp kết nối đầy đủ dưới dạng một mảng một chiều. Tiếp theo, bằng cách khai báo một lớp dense, tất cả các nút trong lớp tiếp theo được kết nối với mọi nút trong lớp trước đó.

![alt text](https://github.com/canozcivelek/traffic-sign-recognition/blob/master/Images/modelSummary.png)

_Tóm tắt mô hình._

### Đào Tạo & Phân Tích
Sau khi định nghĩa mô hình, đã đến lúc đào tạo diễn ra. Mô hình được đào tạo trong 2000 bước cho mỗi epoch, và kích thước epoch được định nghĩa là 10. Đây là lúc tăng tốc phần cứng của Google Colab thực sự tạo ra sự khác biệt vì nó giảm đáng kể thời gian cần thiết để hoàn thành việc đào tạo. Khi việc đào tạo hoàn tất, một số hình ảnh được thực hiện bằng cách vẽ các hàm "Loss" và "Accuracy" của quá trình đào tạo. Phân tích các hàm này là quan trọng để hiểu cách mà việc đào tạo diễn ra. Một người có thể thực hiện các điều chỉnh trên mô hình nếu các đồ thị này cho thấy dấu hiệu của overfitting, underfitting, v.v. Tại đây, độ chính xác kiểm tra trên 97% được thấy, điều này đủ tốt cho mục đích của dự án này.

![alt text](https://github.com/canozcivelek/traffic-sign-recognition/blob/master/Images/graphs.jpg)

_Đồ thị độ chính xác (bên trái) và đồ thị Loss (bên phải)._

### Thử Nghiệm Mô Hình
Cuối cùng, từ một URL, một biển báo giao thông được thử nghiệm để xem liệu mô hình có dự đoán chính xác hay không. Đầu ra được đưa ra dưới dạng nhãn biển báo dự đoán, được định nghĩa trong tệp [signnames.csv](signnames.csv). Dưới đây là hình ảnh của mô hình được cung cấp với biển báo "Giới hạn Tốc độ 30kmh" và dự đoán của nó có thể được xác nhận từ _signnames.csv_ mà giữ nhãn **"1"** cho giới hạn tốc độ 30kmh, chính xác là những gì mô hình đã dự đoán.

![alt text](https://github.com/canozcivelek/traffic-sign-recognition/blob/master/Images/predict.jpg)
---

## Lưu Ý Quan Trọng
Cần lưu ý rằng dự án này được thực hiện cho mục đích giáo dục và tự cải thiện và là một minh họa đơn giản về cách mà các phương pháp học máy có thể được áp dụng hiệu quả để xác định biển báo giao thông. Học máy có thể rất mạnh mẽ và hiệu quả khi nói đến việc phát hiện và phân loại các đối tượng. Dự án này có thể được cải thiện rất nhiều và được nâng cao về mặt hình ảnh, tuy nhiên, trong trạng thái hiện tại, nó được coi là đủ để cho thấy hệ thống hoạt động.

## Lời Cảm Ơn
Trong quá trình thực hiện dự án này, cần phải đề cập đến:
* Rayan Slim - https://github.com/rslim087a vì các bài hướng dẫn của họ về học máy và mạng nơ-ron tích chập.
* Bitbucket - https://bitbucket.org vì các tài nguyên về biển báo giao thông.