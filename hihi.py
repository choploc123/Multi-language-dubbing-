# -*- coding: utf-8 -*-
import jiwer
from underthesea import word_tokenize

def calculate_metrics(reference, hypothesis):
    """
    Tính CER và VWER giữa 2 đoạn văn bản.
    
    Args:
        reference (str): Văn bản gốc (ground truth).
        hypothesis (str): Văn bản được suy luận hoặc kết quả cần đánh giá.

    Returns:
        dict: CER (%) và VWER (%)
    """

    # CER (Character Error Rate)
    cer = jiwer.cer(reference, hypothesis) * 100

    # Tokenize tiếng Việt (có thể dùng word_tokenize của underthesea để chính xác hơn)
    ref_words = word_tokenize(reference, format="text").split()
    hyp_words = word_tokenize(hypothesis, format="text").split()

    # VWER (Vietnamese Word Error Rate)
    vwer = jiwer.wer(" ".join(ref_words), " ".join(hyp_words)) * 100

    return {
        "CER (%)": round(cer, 2),
        "VWER (%)": round(vwer, 2)
    }

# Ví dụ sử dụng
reference_text = "Trong câu chuyện, Thần Gió được mô tả có hình dáng không đầu và có bảo bối là một chiếc quạt mầu nhiệm. Hình tượng kỳ quặc của Thần Gió thể hiện tính khó lường, khó đoán của tự nhiên. Thần Gió có khả năng điều khiển gió, từ việc tạo ra những cơn gió nhẹ cho đến những cơn bão dữ dội. Khả năng này thể hiện sự quyền năng và ảnh hưởng của tự nhiên đối với cuộc sống con người. Việc Thần Gió làm gió nhỏ hay bão lớn, lâu hay mau tùy theo lệnh Ngọc Hoàng phản ánh vai trò của tự nhiên đối với việc sản xuất, thời tiết và sinh kế của người dân."
hypothesis_text = "Trong câu chuyện, thần gió được mô tả có hình dáng không đầu và có bảo bối là một chiếc quạt màu nhẹn mèn trắng. Hình tượng kỳ quạt của thần gió thể hiện tính khó lường, khó đoán của tự nhiên, tràn tràn. Thần gió có khả năng điều khiển gió, từ việc tạo ra những cơn gió nhẹ cho đến những cơn bão sữ sội, nhẹn. Khả năng này thể hiện sự quyền năng và ảnh hưởng của tự nhiên đối với cuộc sống con người. Nóng trắng trong ngờ rắc. Việc thần gió làm gió nhỏ hay bão lớn, lâu hay mau, tùy theo lệnh ngọc hoàng phản ánh vai trò của tự nhiên đối với việc sản xuất, khởi tiết và sinh kế của người dân, nền trắng trong chân."

metrics = calculate_metrics(reference_text, hypothesis_text)
print(metrics)
