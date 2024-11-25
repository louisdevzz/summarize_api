import requests
import json

# Test text with approximately 2000 characters
test_text = """
Metro số 1 (Bến Thành - Suối Tiên) dự tính khai thác ngày 22/12, với mức 6.000-20.000 đồng/lượt, vé tháng 300.000 đồng, ngoài ra còn vé 1-3 ngày, không giới hạn lượt đi. Thông tin được Ban Quản lý đường sắt đô thị TP HCM (chủ đầu tư) nêu tại họp báo kinh tế và xã hội thành phố diễn ra chiều 21/11. Theo chủ đầu tư, hiện dự án đã hoàn thành 100% khối lượng thi công, đang trong thời gian làm các thủ tục để hoàn thành và đưa vào vận hành thương mại.

Giá vé của tuyến Metro số 1 đã được UBND TP HCM thông qua. Khách đi tuyến tàu điện đầu tiên ở thành phố có thể chọn các loại vé, gồm theo lượt, một ngày, ba ngày và theo tháng. Trong đó, khách mua vé lượt nếu dùng tiền mặt sẽ trả 7.000 đồng đến 20.000 đồng, tùy quãng đường. Nếu chọn thanh toán không tiền mặt, vé lượt áp dụng 6.000-19.000 đồng.

Đối với vé tháng, mức giá áp dụng 300.000 đồng mỗi khách; học sinh, sinh viên được giảm 50%, còn 150.000 đồng/tháng. Ngoài các loại vé trên, khách có thể mua vé một ngày hoặc ba ngày, lần lượt 40.000 đồng và 90.000 đồng. Các loại vé này không giới hạn lượt đi. Ngoài ra, một số hành khách như người khuyết tật, cao tuổi... sẽ được miễn vé theo chính sách của TP HCM.

Thành phố dự kiến phát hành hơn 2 triệu thẻ đi Metro số 1 trong giai đoạn đầu, mỗi ngày ước tính phục vụ gần 40.000 khách. Ngoài phương án giá vé, trong 30 ngày đầu khai thác thương mại, khách đi Metro Bến Thành - Suối Tiên cùng 17 tuyến buýt kết nối sẽ được miễn vé nhằm khuyến khích người dân sử dụng tuyến tàu điện đầu tiên ở địa bàn.

Tuyến Metro số 1 có tổng chiều dài gần 20 km, đi qua các quận 1, Bình Thạnh, Thủ Đức (TP HCM) và thị xã Dĩ An (Bình Dương). Tuyến tàu có 14 nhà ga (3 ga ngầm, 11 ga trên cao). Đoàn tàu có thể chạy với tốc độ tối đa 110 km/h ở đoạn trên cao và 80 km/h ở đoạn ngầm. Thời gian di chuyển từ Bến Thành đến Suối Tiên khoảng 30 phút.

Dự án có tổng mức đầu tư hơn 43.700 tỷ đồng (tương đương gần 2 tỷ USD), được khởi công từ tháng 8/2012. Đây là tuyến metro đầu tiên của TP HCM, được kỳ vọng góp phần giảm ùn tắc giao thông và phát triển hệ thống vận tải hành khách công cộng của thành phố.
"""

def test_summarize_api():
    url = "http://localhost:8000/summarize"
    
    payload = {
        "text": test_text,
        "max_length": 500,
        "min_length": 100
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        result = response.json()
        
        print("\nAPI Test Results:")
        print(f"\nOriginal text length: {result['original_length']}")
        print(f"\nSummary text: {result['summary']}")
        print(f"\nSummary text length: {result['summary_length']}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None

if __name__ == "__main__":
    test_summarize_api() 