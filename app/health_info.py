"""
Module xử lý thông tin sức khỏe và gợi ý dựa trên kết quả detection và classification
"""

# Mapping các vấn đề thẩm mỹ/thông thường (không cần classification)
COSMETIC_ISSUES = {
    "Dark Circle": {
        "name_vi": "Quầng thâm mắt",
        "description": "Vùng da tối màu quanh mắt, thường do thiếu ngủ, di truyền hoặc lão hóa",
        "lifestyle": [
            "Ngủ đủ 7-8 giờ mỗi đêm và duy trì giờ ngủ đều đặn",
            "Sử dụng kem dưỡng mắt chứa retinol, vitamin C, hoặc peptide",
            "Chườm lạnh vùng mắt 10-15 phút mỗi ngày để giảm sưng",
            "Tránh dụi mắt và chạm tay lên vùng mắt",
            "Bảo vệ mắt khỏi ánh nắng mặt trời bằng kính râm và kem chống nắng",
            "Ngủ với gối cao hơn để tránh tích tụ chất lỏng",
            "Massage nhẹ nhàng vùng mắt mỗi sáng để cải thiện tuần hoàn",
            "Tránh thức khuya và làm việc quá nhiều với màn hình",
            "Sử dụng mặt nạ mắt chứa caffeine hoặc vitamin K",
            "Tẩy trang sạch sẽ trước khi ngủ",
            "Tránh hút thuốc lá vì làm giảm lưu thông máu",
            "Quản lý căng thẳng và tập thể dục thường xuyên"
        ],
        "diet": [
            "Tăng cường thực phẩm giàu vitamin K (rau xanh đậm, bông cải xanh, cải bắp)",
            "Bổ sung sắt từ thực phẩm (thịt đỏ, đậu, rau bina, đậu lăng)",
            "Uống đủ nước (2-3 lít/ngày) để giữ ẩm da",
            "Hạn chế muối để tránh giữ nước và sưng",
            "Bổ sung vitamin C (cam, ổi, ớt chuông) và E (các loại hạt)",
            "Tăng cường thực phẩm giàu omega-3 (cá hồi, quả óc chó)",
            "Bổ sung collagen từ nước hầm xương hoặc thực phẩm bổ sung",
            "Hạn chế rượu bia và caffeine quá mức",
            "Tăng cường thực phẩm giàu chất chống oxy hóa (quả mọng, trà xanh)",
            "Bổ sung kẽm từ thực phẩm (hàu, thịt đỏ, các loại hạt)",
            "Tăng cường vitamin A từ cà rốt, khoai lang",
            "Uống trà xanh hoặc trà hoa cúc để chống viêm"
        ]
    },
    "Dark circle": {
        "name_vi": "Quầng thâm mắt",
        "description": "Vùng da tối màu quanh mắt, thường do thiếu ngủ, di truyền hoặc lão hóa",
        "lifestyle": [
            "Ngủ đủ 7-8 giờ mỗi đêm và duy trì giờ ngủ đều đặn",
            "Sử dụng kem dưỡng mắt chứa retinol, vitamin C, hoặc peptide",
            "Chườm lạnh vùng mắt 10-15 phút mỗi ngày để giảm sưng",
            "Tránh dụi mắt và chạm tay lên vùng mắt",
            "Bảo vệ mắt khỏi ánh nắng mặt trời bằng kính râm và kem chống nắng",
            "Ngủ với gối cao hơn để tránh tích tụ chất lỏng",
            "Massage nhẹ nhàng vùng mắt mỗi sáng để cải thiện tuần hoàn",
            "Tránh thức khuya và làm việc quá nhiều với màn hình",
            "Sử dụng mặt nạ mắt chứa caffeine hoặc vitamin K",
            "Tẩy trang sạch sẽ trước khi ngủ",
            "Tránh hút thuốc lá vì làm giảm lưu thông máu",
            "Quản lý căng thẳng và tập thể dục thường xuyên"
        ],
        "diet": [
            "Tăng cường thực phẩm giàu vitamin K (rau xanh đậm, bông cải xanh, cải bắp)",
            "Bổ sung sắt từ thực phẩm (thịt đỏ, đậu, rau bina, đậu lăng)",
            "Uống đủ nước (2-3 lít/ngày) để giữ ẩm da",
            "Hạn chế muối để tránh giữ nước và sưng",
            "Bổ sung vitamin C (cam, ổi, ớt chuông) và E (các loại hạt)",
            "Tăng cường thực phẩm giàu omega-3 (cá hồi, quả óc chó)",
            "Bổ sung collagen từ nước hầm xương hoặc thực phẩm bổ sung",
            "Hạn chế rượu bia và caffeine quá mức",
            "Tăng cường thực phẩm giàu chất chống oxy hóa (quả mọng, trà xanh)",
            "Bổ sung kẽm từ thực phẩm (hàu, thịt đỏ, các loại hạt)",
            "Tăng cường vitamin A từ cà rốt, khoai lang",
            "Uống trà xanh hoặc trà hoa cúc để chống viêm"
        ]
    },
    "Eyebag": {
        "name_vi": "Bọng mắt",
        "description": "Túi phồng dưới mắt, thường do tích tụ chất lỏng hoặc mỡ",
        "lifestyle": [
            "Ngủ đủ giấc (7-8 giờ) và ngủ đúng tư thế (đầu cao hơn)",
            "Chườm lạnh vùng mắt vào buổi sáng 10-15 phút",
            "Massage nhẹ nhàng vùng mắt theo chiều từ trong ra ngoài",
            "Sử dụng kem dưỡng mắt có caffeine hoặc peptide",
            "Tránh uống nhiều nước trước khi ngủ (2-3 giờ trước)",
            "Ngủ với gối cao để tránh tích tụ chất lỏng",
            "Tránh dụi mắt và chạm tay lên vùng mắt",
            "Sử dụng mặt nạ mắt làm mát",
            "Tránh thức khuya và làm việc quá nhiều với màn hình",
            "Tập thể dục thường xuyên để cải thiện tuần hoàn",
            "Tránh hút thuốc lá và rượu bia",
            "Quản lý căng thẳng và ngủ đủ giấc"
        ],
        "diet": [
            "Hạn chế muối để tránh giữ nước và sưng",
            "Uống đủ nước trong ngày (2-3 lít) nhưng giảm vào buổi tối",
            "Tăng cường thực phẩm giàu kali (chuối, khoai tây, bơ, rau bina)",
            "Tránh rượu bia và đồ uống có cồn",
            "Bổ sung collagen từ nước hầm xương hoặc thực phẩm bổ sung",
            "Tăng cường thực phẩm giàu vitamin C (cam, ổi, ớt chuông)",
            "Bổ sung omega-3 từ cá và các loại hạt",
            "Hạn chế thực phẩm chế biến sẵn và đồ ăn nhanh",
            "Tăng cường thực phẩm giàu chất chống oxy hóa (quả mọng, trà xanh)",
            "Bổ sung kẽm từ thực phẩm (hàu, thịt đỏ, các loại hạt)",
            "Tăng cường vitamin E từ các loại hạt và dầu thực vật",
            "Uống trà xanh hoặc trà hoa cúc để giảm viêm"
        ]
    },
    "blackhead": {
        "name_vi": "Mụn đầu đen",
        "description": "Lỗ chân lông bị tắc nghẽn bởi bã nhờn và tế bào chết",
        "lifestyle": [
            "Rửa mặt 2 lần/ngày với sữa rửa mặt dịu nhẹ, pH cân bằng",
            "Tẩy tế bào chết 1-2 lần/tuần với sản phẩm dịu nhẹ",
            "Sử dụng sản phẩm chứa salicylic acid (BHA) hoặc benzoyl peroxide",
            "Tránh nặn mụn bằng tay để tránh nhiễm trùng và sẹo",
            "Tẩy trang sạch sẽ trước khi ngủ, kể cả khi không trang điểm",
            "Sử dụng mặt nạ đất sét 1-2 lần/tuần để làm sạch lỗ chân lông",
            "Tránh chạm tay lên mặt, đặc biệt khi tay chưa rửa sạch",
            "Thay vỏ gối thường xuyên (2-3 lần/tuần)",
            "Tránh sử dụng mỹ phẩm gây bít tắc lỗ chân lông (non-comedogenic)",
            "Sử dụng toner chứa axit để làm sạch sâu",
            "Tránh stress quá mức vì có thể làm tăng sản xuất bã nhờn",
            "Tập thể dục thường xuyên và tắm ngay sau khi tập"
        ],
        "diet": [
            "Hạn chế thực phẩm dầu mỡ và đồ ăn nhanh",
            "Tăng cường rau xanh và trái cây giàu chất chống oxy hóa",
            "Uống đủ nước (2-3 lít/ngày) để thanh lọc cơ thể",
            "Hạn chế đồ ngọt và thực phẩm có chỉ số đường huyết cao",
            "Bổ sung omega-3 từ cá hồi, quả óc chó, hạt chia",
            "Tăng cường kẽm từ thực phẩm (hàu, thịt đỏ, các loại hạt)",
            "Bổ sung vitamin A từ cà rốt, khoai lang, rau bina",
            "Hạn chế sữa và sản phẩm từ sữa, đặc biệt là sữa tách béo",
            "Bổ sung probiotic từ sữa chua, kefir để cân bằng hệ vi sinh",
            "Tăng cường thực phẩm giàu vitamin E (hạnh nhân, bơ)",
            "Uống trà xanh để chống viêm và chống oxy hóa",
            "Tránh thức ăn cay nóng quá mức"
        ]
    },
    "blackheads": {
        "name_vi": "Mụn đầu đen",
        "description": "Lỗ chân lông bị tắc nghẽn bởi bã nhờn và tế bào chết",
        "lifestyle": [
            "Rửa mặt 2 lần/ngày với sữa rửa mặt dịu nhẹ, pH cân bằng",
            "Tẩy tế bào chết 1-2 lần/tuần với sản phẩm dịu nhẹ",
            "Sử dụng sản phẩm chứa salicylic acid (BHA) hoặc benzoyl peroxide",
            "Tránh nặn mụn bằng tay để tránh nhiễm trùng và sẹo",
            "Tẩy trang sạch sẽ trước khi ngủ, kể cả khi không trang điểm",
            "Sử dụng mặt nạ đất sét 1-2 lần/tuần để làm sạch lỗ chân lông",
            "Tránh chạm tay lên mặt, đặc biệt khi tay chưa rửa sạch",
            "Thay vỏ gối thường xuyên (2-3 lần/tuần)",
            "Tránh sử dụng mỹ phẩm gây bít tắc lỗ chân lông (non-comedogenic)",
            "Sử dụng toner chứa axit để làm sạch sâu",
            "Tránh stress quá mức vì có thể làm tăng sản xuất bã nhờn",
            "Tập thể dục thường xuyên và tắm ngay sau khi tập"
        ],
        "diet": [
            "Hạn chế thực phẩm dầu mỡ và đồ ăn nhanh",
            "Tăng cường rau xanh và trái cây giàu chất chống oxy hóa",
            "Uống đủ nước (2-3 lít/ngày) để thanh lọc cơ thể",
            "Hạn chế đồ ngọt và thực phẩm có chỉ số đường huyết cao",
            "Bổ sung omega-3 từ cá hồi, quả óc chó, hạt chia",
            "Tăng cường kẽm từ thực phẩm (hàu, thịt đỏ, các loại hạt)",
            "Bổ sung vitamin A từ cà rốt, khoai lang, rau bina",
            "Hạn chế sữa và sản phẩm từ sữa, đặc biệt là sữa tách béo",
            "Bổ sung probiotic từ sữa chua, kefir để cân bằng hệ vi sinh",
            "Tăng cường thực phẩm giàu vitamin E (hạnh nhân, bơ)",
            "Uống trà xanh để chống viêm và chống oxy hóa",
            "Tránh thức ăn cay nóng quá mức"
        ]
    },
    "dark spot": {
        "name_vi": "Đốm nâu/Thâm nám",
        "description": "Vùng da sẫm màu do tăng sắc tố, thường do ánh nắng hoặc viêm",
        "lifestyle": [
            "Sử dụng kem chống nắng SPF 30+ hàng ngày, bôi lại sau mỗi 2 giờ",
            "Tránh tiếp xúc trực tiếp với ánh nắng mặt trời, đặc biệt từ 10h-16h",
            "Sử dụng sản phẩm chứa vitamin C, niacinamide, hoặc retinol",
            "Tẩy tế bào chết nhẹ nhàng 1-2 lần/tuần",
            "Mặc quần áo bảo vệ khi ra ngoài (áo dài tay, mũ rộng vành)",
            "Sử dụng serum chứa axit kojic hoặc arbutin",
            "Tránh nặn mụn để tránh để lại thâm",
            "Dưỡng ẩm da đầy đủ để phục hồi hàng rào da",
            "Tránh sử dụng sản phẩm gây kích ứng",
            "Sử dụng mặt nạ làm sáng da chứa vitamin C",
            "Tránh hút thuốc lá vì làm tăng sắc tố",
            "Quản lý căng thẳng và ngủ đủ giấc"
        ],
        "diet": [
            "Tăng cường thực phẩm giàu vitamin C (cam, ổi, kiwi, ớt chuông)",
            "Bổ sung chất chống oxy hóa (quả mọng, cà chua, trà xanh)",
            "Uống đủ nước (2-3 lít/ngày) để giữ ẩm da",
            "Hạn chế thực phẩm chế biến sẵn và đồ ăn nhanh",
            "Bổ sung vitamin E từ các loại hạt và dầu thực vật",
            "Tăng cường thực phẩm giàu lycopene (cà chua nấu chín, dưa hấu)",
            "Bổ sung omega-3 từ cá và các loại hạt",
            "Tăng cường thực phẩm giàu beta-carotene (cà rốt, khoai lang)",
            "Bổ sung kẽm từ thực phẩm (hàu, thịt đỏ, các loại hạt)",
            "Uống trà xanh thường xuyên để chống oxy hóa",
            "Hạn chế đường và thực phẩm có chỉ số đường huyết cao",
            "Tăng cường thực phẩm giàu resveratrol (nho đỏ, rượu vang đỏ)"
        ]
    },
    "darkspot": {
        "name_vi": "Đốm nâu/Thâm nám",
        "description": "Vùng da sẫm màu do tăng sắc tố, thường do ánh nắng hoặc viêm",
        "lifestyle": [
            "Sử dụng kem chống nắng SPF 30+ hàng ngày, bôi lại sau mỗi 2 giờ",
            "Tránh tiếp xúc trực tiếp với ánh nắng mặt trời, đặc biệt từ 10h-16h",
            "Sử dụng sản phẩm chứa vitamin C, niacinamide, hoặc retinol",
            "Tẩy tế bào chết nhẹ nhàng 1-2 lần/tuần",
            "Mặc quần áo bảo vệ khi ra ngoài (áo dài tay, mũ rộng vành)",
            "Sử dụng serum chứa axit kojic hoặc arbutin",
            "Tránh nặn mụn để tránh để lại thâm",
            "Dưỡng ẩm da đầy đủ để phục hồi hàng rào da",
            "Tránh sử dụng sản phẩm gây kích ứng",
            "Sử dụng mặt nạ làm sáng da chứa vitamin C",
            "Tránh hút thuốc lá vì làm tăng sắc tố",
            "Quản lý căng thẳng và ngủ đủ giấc"
        ],
        "diet": [
            "Tăng cường thực phẩm giàu vitamin C (cam, ổi, kiwi, ớt chuông)",
            "Bổ sung chất chống oxy hóa (quả mọng, cà chua, trà xanh)",
            "Uống đủ nước (2-3 lít/ngày) để giữ ẩm da",
            "Hạn chế thực phẩm chế biến sẵn và đồ ăn nhanh",
            "Bổ sung vitamin E từ các loại hạt và dầu thực vật",
            "Tăng cường thực phẩm giàu lycopene (cà chua nấu chín, dưa hấu)",
            "Bổ sung omega-3 từ cá và các loại hạt",
            "Tăng cường thực phẩm giàu beta-carotene (cà rốt, khoai lang)",
            "Bổ sung kẽm từ thực phẩm (hàu, thịt đỏ, các loại hạt)",
            "Uống trà xanh thường xuyên để chống oxy hóa",
            "Hạn chế đường và thực phẩm có chỉ số đường huyết cao",
            "Tăng cường thực phẩm giàu resveratrol (nho đỏ, rượu vang đỏ)"
        ]
    },
    "freckle": {
        "name_vi": "Tàn nhang",
        "description": "Các đốm nhỏ sẫm màu trên da, thường do di truyền và ánh nắng",
        "lifestyle": [
            "Sử dụng kem chống nắng SPF 30+ hàng ngày, bôi lại thường xuyên",
            "Tránh tiếp xúc trực tiếp với ánh nắng mặt trời, đặc biệt từ 10h-16h",
            "Sử dụng sản phẩm chứa vitamin C, niacinamide để làm sáng da",
            "Mặc quần áo bảo vệ khi ra ngoài (áo dài tay, mũ rộng vành)",
            "Đội mũ rộng vành và đeo kính râm",
            "Sử dụng serum chứa axit kojic hoặc arbutin",
            "Tẩy tế bào chết nhẹ nhàng 1-2 lần/tuần",
            "Dưỡng ẩm da đầy đủ để phục hồi hàng rào da",
            "Tránh sử dụng giường tắm nắng (tanning bed)",
            "Sử dụng mặt nạ làm sáng da chứa vitamin C",
            "Tránh hút thuốc lá vì làm tăng sắc tố",
            "Quản lý căng thẳng và ngủ đủ giấc"
        ],
        "diet": [
            "Tăng cường thực phẩm giàu vitamin C (cam, ổi, kiwi, ớt chuông)",
            "Bổ sung chất chống oxy hóa (quả mọng, cà chua, trà xanh)",
            "Uống đủ nước (2-3 lít/ngày) để giữ ẩm da",
            "Hạn chế thực phẩm chế biến sẵn và đồ ăn nhanh",
            "Bổ sung vitamin E (các loại hạt, dầu thực vật) và beta-carotene (cà rốt, khoai lang)",
            "Tăng cường thực phẩm giàu lycopene (cà chua nấu chín, dưa hấu)",
            "Bổ sung omega-3 từ cá và các loại hạt",
            "Tăng cường thực phẩm giàu selen (các loại hạt Brazil, cá, trứng)",
            "Uống trà xanh thường xuyên để chống oxy hóa",
            "Bổ sung kẽm từ thực phẩm (hàu, thịt đỏ, các loại hạt)",
            "Hạn chế đường và thực phẩm có chỉ số đường huyết cao",
            "Tăng cường thực phẩm giàu resveratrol (nho đỏ, rượu vang đỏ)"
        ]
    },
    "whitehead": {
        "name_vi": "Mụn đầu trắng",
        "description": "Lỗ chân lông bị tắc nghẽn và đóng kín bởi bã nhờn",
        "lifestyle": [
            "Rửa mặt 2 lần/ngày với sữa rửa mặt dịu nhẹ, pH cân bằng",
            "Tẩy tế bào chết 1-2 lần/tuần với sản phẩm chứa BHA",
            "Sử dụng sản phẩm chứa salicylic acid (BHA) hoặc benzoyl peroxide",
            "Tránh nặn mụn bằng tay để tránh nhiễm trùng và sẹo",
            "Tẩy trang sạch sẽ trước khi ngủ, kể cả khi không trang điểm",
            "Sử dụng mặt nạ đất sét 1-2 lần/tuần để làm sạch lỗ chân lông",
            "Tránh chạm tay lên mặt, đặc biệt khi tay chưa rửa sạch",
            "Thay vỏ gối thường xuyên (2-3 lần/tuần)",
            "Tránh sử dụng mỹ phẩm gây bít tắc lỗ chân lông (non-comedogenic)",
            "Sử dụng toner chứa axit để làm sạch sâu",
            "Tránh stress quá mức vì có thể làm tăng sản xuất bã nhờn",
            "Tập thể dục thường xuyên và tắm ngay sau khi tập"
        ],
        "diet": [
            "Hạn chế thực phẩm dầu mỡ và đồ ăn nhanh",
            "Tăng cường rau xanh và trái cây giàu chất chống oxy hóa",
            "Uống đủ nước (2-3 lít/ngày) để thanh lọc cơ thể",
            "Hạn chế đồ ngọt và sữa, đặc biệt là sữa tách béo",
            "Bổ sung omega-3 từ cá hồi, quả óc chó, hạt chia",
            "Tăng cường kẽm từ thực phẩm (hàu, thịt đỏ, các loại hạt)",
            "Bổ sung vitamin A từ cà rốt, khoai lang, rau bina",
            "Hạn chế thực phẩm có chỉ số đường huyết cao",
            "Bổ sung probiotic từ sữa chua, kefir để cân bằng hệ vi sinh",
            "Tăng cường thực phẩm giàu vitamin E (hạnh nhân, bơ)",
            "Uống trà xanh để chống viêm và chống oxy hóa",
            "Tránh thức ăn cay nóng quá mức"
        ]
    },
    "whiteheads": {
        "name_vi": "Mụn đầu trắng",
        "description": "Lỗ chân lông bị tắc nghẽn và đóng kín bởi bã nhờn",
        "lifestyle": [
            "Rửa mặt 2 lần/ngày với sữa rửa mặt dịu nhẹ, pH cân bằng",
            "Tẩy tế bào chết 1-2 lần/tuần với sản phẩm chứa BHA",
            "Sử dụng sản phẩm chứa salicylic acid (BHA) hoặc benzoyl peroxide",
            "Tránh nặn mụn bằng tay để tránh nhiễm trùng và sẹo",
            "Tẩy trang sạch sẽ trước khi ngủ, kể cả khi không trang điểm",
            "Sử dụng mặt nạ đất sét 1-2 lần/tuần để làm sạch lỗ chân lông",
            "Tránh chạm tay lên mặt, đặc biệt khi tay chưa rửa sạch",
            "Thay vỏ gối thường xuyên (2-3 lần/tuần)",
            "Tránh sử dụng mỹ phẩm gây bít tắc lỗ chân lông (non-comedogenic)",
            "Sử dụng toner chứa axit để làm sạch sâu",
            "Tránh stress quá mức vì có thể làm tăng sản xuất bã nhờn",
            "Tập thể dục thường xuyên và tắm ngay sau khi tập"
        ],
        "diet": [
            "Hạn chế thực phẩm dầu mỡ và đồ ăn nhanh",
            "Tăng cường rau xanh và trái cây giàu chất chống oxy hóa",
            "Uống đủ nước (2-3 lít/ngày) để thanh lọc cơ thể",
            "Hạn chế đồ ngọt và sữa, đặc biệt là sữa tách béo",
            "Bổ sung omega-3 từ cá hồi, quả óc chó, hạt chia",
            "Tăng cường kẽm từ thực phẩm (hàu, thịt đỏ, các loại hạt)",
            "Bổ sung vitamin A từ cà rốt, khoai lang, rau bina",
            "Hạn chế thực phẩm có chỉ số đường huyết cao",
            "Bổ sung probiotic từ sữa chua, kefir để cân bằng hệ vi sinh",
            "Tăng cường thực phẩm giàu vitamin E (hạnh nhân, bơ)",
            "Uống trà xanh để chống viêm và chống oxy hóa",
            "Tránh thức ăn cay nóng quá mức"
        ]
    },
    "wrinkle": {
        "name_vi": "Nếp nhăn",
        "description": "Các đường nếp trên da do lão hóa, ánh nắng và biểu cảm",
        "lifestyle": [
            "Sử dụng kem chống nắng SPF 30+ hàng ngày, bôi lại thường xuyên",
            "Dưỡng ẩm da đầy đủ, đặc biệt sau khi tắm",
            "Sử dụng sản phẩm chứa retinol, peptide, hoặc axit hyaluronic",
            "Tránh hút thuốc lá và khói thuốc thụ động",
            "Ngủ đủ giấc (7-8 giờ/đêm) và quản lý căng thẳng",
            "Tránh tiếp xúc trực tiếp với ánh nắng mặt trời, đặc biệt từ 10h-16h",
            "Mặc quần áo bảo vệ khi ra ngoài (áo dài tay, mũ rộng vành)",
            "Sử dụng serum chứa vitamin C vào buổi sáng",
            "Tẩy trang sạch sẽ trước khi ngủ",
            "Tránh rửa mặt quá nhiều lần và sử dụng nước quá nóng",
            "Sử dụng máy tạo độ ẩm trong phòng ngủ",
            "Tập thể dục thường xuyên để cải thiện tuần hoàn máu"
        ],
        "diet": [
            "Tăng cường thực phẩm giàu chất chống oxy hóa (quả mọng, trà xanh, cà chua)",
            "Bổ sung collagen từ nước hầm xương, cá, hoặc thực phẩm bổ sung",
            "Tăng cường vitamin C (cam, ổi, ớt chuông) và E (các loại hạt, dầu thực vật)",
            "Uống đủ nước (2-3 lít/ngày) để giữ ẩm da từ bên trong",
            "Hạn chế đường và thực phẩm chế biến sẵn",
            "Tăng cường omega-3 từ cá hồi, quả óc chó, hạt chia",
            "Bổ sung kẽm từ thực phẩm (hàu, thịt đỏ, các loại hạt)",
            "Tăng cường thực phẩm giàu beta-carotene (cà rốt, khoai lang, bí đỏ)",
            "Bổ sung selen từ các loại hạt Brazil, cá, trứng",
            "Uống trà xanh thường xuyên để chống oxy hóa",
            "Tăng cường thực phẩm giàu lycopene (cà chua nấu chín, dưa hấu)",
            "Bổ sung đầy đủ protein từ thịt nạc, cá, đậu để sản xuất collagen"
        ]
    }
}

# Mapping các bệnh với thông tin và gợi ý
DISEASE_INFO = {
    "acne": {
        "name_vi": "Mụn trứng cá",
        "description": "Tình trạng viêm da do tắc nghẽn lỗ chân lông",
        "lifestyle": [
            "Rửa mặt 2 lần/ngày với sữa rửa mặt dịu nhẹ, pH cân bằng",
            "Tránh chạm tay lên mặt, đặc biệt khi tay chưa rửa sạch",
            "Giữ da sạch và khô ráo, tránh để da quá ẩm ướt",
            "Tránh sử dụng mỹ phẩm gây bít tắc lỗ chân lông (non-comedogenic)",
            "Thay vỏ gối thường xuyên (2-3 lần/tuần) để tránh vi khuẩn",
            "Sử dụng sản phẩm chứa benzoyl peroxide hoặc salicylic acid",
            "Tránh nặn mụn bằng tay để tránh nhiễm trùng và sẹo",
            "Tẩy trang sạch sẽ trước khi ngủ, kể cả khi không trang điểm",
            "Tránh stress quá mức vì có thể làm tăng sản xuất hormone gây mụn",
            "Tập thể dục thường xuyên để cải thiện tuần hoàn máu",
            "Ngủ đủ 7-8 giờ mỗi đêm để da có thời gian phục hồi",
            "Tránh tiếp xúc với khói bụi và môi trường ô nhiễm"
        ],
        "diet": [
            "Hạn chế thực phẩm có chỉ số đường huyết cao (đồ ngọt, bánh mì trắng, gạo trắng)",
            "Giảm tiêu thụ sữa và sản phẩm từ sữa, đặc biệt là sữa tách béo",
            "Tăng cường thực phẩm giàu omega-3 (cá hồi, cá thu, quả óc chó, hạt chia)",
            "Ăn nhiều rau xanh và trái cây tươi giàu chất chống oxy hóa",
            "Uống đủ nước (2-3 lít/ngày) để thanh lọc cơ thể",
            "Bổ sung kẽm từ thực phẩm (hàu, thịt đỏ, các loại hạt)",
            "Tăng cường vitamin A từ cà rốt, khoai lang, rau bina",
            "Hạn chế đồ ăn cay nóng và nhiều dầu mỡ",
            "Bổ sung probiotic từ sữa chua, kefir để cân bằng hệ vi sinh đường ruột",
            "Tránh thức ăn nhanh và thực phẩm chế biến sẵn",
            "Tăng cường thực phẩm giàu vitamin E (hạnh nhân, bơ, hạt hướng dương)",
            "Uống trà xanh để chống viêm và chống oxy hóa"
        ]
    },
    "carcinoma": {
        "name_vi": "Ung thư biểu mô",
        "description": "Tình trạng nghiêm trọng cần được bác sĩ chuyên khoa đánh giá",
        "lifestyle": [
            "Tránh tiếp xúc trực tiếp với ánh nắng mặt trời, đặc biệt từ 10h-16h",
            "Sử dụng kem chống nắng SPF 30+ hàng ngày, bôi lại sau mỗi 2 giờ",
            "Mặc quần áo bảo vệ khi ra ngoài (áo dài tay, mũ rộng vành)",
            "Không hút thuốc lá và tránh khói thuốc thụ động",
            "Khám da định kỳ với bác sĩ da liễu ít nhất 1 lần/năm",
            "Tự kiểm tra da hàng tháng để phát hiện thay đổi bất thường",
            "Tránh sử dụng giường tắm nắng (tanning bed)",
            "Bảo vệ môi bằng son dưỡng có SPF",
            "Kiểm tra các nốt ruồi và đốm da mới xuất hiện",
            "Tránh tiếp xúc với hóa chất độc hại và bức xạ",
            "Duy trì cân nặng hợp lý và tập thể dục thường xuyên",
            "Quản lý căng thẳng và ngủ đủ giấc để tăng cường miễn dịch"
        ],
        "diet": [
            "Tăng cường thực phẩm giàu chất chống oxy hóa (quả mọng, cà chua, rau xanh đậm, trà xanh)",
            "Bổ sung vitamin D từ thực phẩm (cá béo, lòng đỏ trứng) hoặc bác sĩ tư vấn",
            "Hạn chế thực phẩm chế biến sẵn và thịt đỏ",
            "Ăn nhiều cá giàu omega-3 (cá hồi, cá thu, cá mòi)",
            "Tránh rượu bia và đồ uống có cồn",
            "Tăng cường thực phẩm giàu beta-carotene (cà rốt, khoai lang, bí đỏ)",
            "Bổ sung selen từ các loại hạt Brazil, cá, trứng",
            "Tăng cường thực phẩm giàu lycopene (cà chua nấu chín, dưa hấu)",
            "Uống trà xanh thường xuyên để chống oxy hóa",
            "Hạn chế đường và thực phẩm có chỉ số đường huyết cao",
            "Bổ sung đầy đủ vitamin C và E từ thực phẩm tự nhiên",
            "Ăn nhiều rau họ cải (bông cải xanh, cải bắp) để tăng cường chất chống ung thư"
        ]
    },
    "eczema": {
        "name_vi": "Viêm da dị ứng (Eczema)",
        "description": "Tình trạng viêm da mãn tính gây ngứa và khô da",
        "lifestyle": [
            "Dưỡng ẩm da 2-3 lần/ngày, đặc biệt sau khi tắm trong vòng 3 phút",
            "Tắm nước ấm (không quá nóng), thời gian tắm ngắn (5-10 phút)",
            "Sử dụng sữa tắm không chứa xà phòng, pH cân bằng",
            "Tránh các chất gây kích ứng (hóa chất, nước hoa, chất tẩy rửa mạnh)",
            "Mặc quần áo cotton mềm mại, tránh vải len hoặc sợi tổng hợp",
            "Giữ nhiệt độ phòng mát mẻ, tránh quá nóng hoặc quá lạnh",
            "Tránh gãi hoặc cọ xát mạnh lên vùng da bị ảnh hưởng",
            "Sử dụng máy tạo độ ẩm trong phòng ngủ",
            "Tránh stress và học cách quản lý căng thẳng",
            "Ngủ đủ giấc để da có thời gian phục hồi",
            "Tránh tiếp xúc với các chất gây dị ứng đã biết",
            "Sử dụng kem dưỡng có chứa ceramide để phục hồi hàng rào da"
        ],
        "diet": [
            "Tránh thực phẩm gây dị ứng đã biết (trứng, sữa, đậu phộng, hải sản)",
            "Tăng cường thực phẩm giàu omega-3 (cá hồi, quả óc chó, hạt lanh)",
            "Bổ sung probiotic (sữa chua, kefir, kimchi) để cân bằng hệ vi sinh",
            "Hạn chế thực phẩm chế biến sẵn và phụ gia thực phẩm",
            "Uống đủ nước (2-3 lít/ngày) để giữ ẩm da từ bên trong",
            "Tăng cường thực phẩm giàu quercetin (táo, hành tây, quả mọng)",
            "Bổ sung vitamin D từ thực phẩm hoặc ánh nắng nhẹ",
            "Hạn chế đường và thực phẩm có chỉ số đường huyết cao",
            "Tăng cường thực phẩm giàu kẽm (hàu, thịt đỏ, các loại hạt)",
            "Bổ sung vitamin E từ các loại hạt và dầu thực vật",
            "Uống trà xanh để chống viêm",
            "Tăng cường thực phẩm giàu vitamin C (cam, ổi, ớt chuông) để tăng collagen"
        ]
    },
    "keratosis": {
        "name_vi": "Dày sừng",
        "description": "Tình trạng da dày lên do tích tụ keratin",
        "lifestyle": [
            "Tẩy tế bào chết nhẹ nhàng 1-2 lần/tuần với sản phẩm dịu nhẹ",
            "Dưỡng ẩm da thường xuyên, đặc biệt sau khi tắm",
            "Tránh cọ xát mạnh lên da, massage nhẹ nhàng",
            "Sử dụng kem dưỡng có chứa axit salicylic hoặc urea (10-20%)",
            "Bảo vệ da khỏi ánh nắng mặt trời bằng kem chống nắng SPF 30+",
            "Tắm nước ấm, không quá nóng để tránh làm khô da",
            "Sử dụng sữa tắm dịu nhẹ, không chứa xà phòng",
            "Tránh mặc quần áo chật, cọ xát vào vùng da bị ảnh hưởng",
            "Giữ da sạch sẽ nhưng không rửa quá nhiều lần",
            "Sử dụng máy tạo độ ẩm trong phòng ngủ",
            "Tránh stress và ngủ đủ giấc",
            "Khám bác sĩ da liễu nếu tình trạng không cải thiện"
        ],
        "diet": [
            "Tăng cường vitamin A (cà rốt, khoai lang, rau xanh đậm, bí đỏ)",
            "Bổ sung vitamin E (các loại hạt, dầu thực vật, bơ)",
            "Uống đủ nước (2-3 lít/ngày) để giữ ẩm da",
            "Hạn chế thực phẩm chế biến sẵn và đồ ăn nhanh",
            "Ăn nhiều trái cây và rau củ giàu chất chống oxy hóa",
            "Tăng cường omega-3 từ cá và các loại hạt",
            "Bổ sung kẽm từ thực phẩm (hàu, thịt đỏ, các loại hạt)",
            "Hạn chế đường và thực phẩm có chỉ số đường huyết cao",
            "Tăng cường thực phẩm giàu biotin (trứng, các loại hạt, cá)",
            "Bổ sung vitamin C từ cam, ổi, ớt chuông",
            "Uống trà xanh để chống oxy hóa",
            "Tăng cường thực phẩm giàu collagen (nước hầm xương, cá)"
        ]
    },
    "rosacea": {
        "name_vi": "Đỏ mặt (Rosacea)",
        "description": "Tình trạng da đỏ và viêm, thường ở vùng mặt",
        "lifestyle": [
            "Tránh các yếu tố kích thích (nhiệt độ cao, ánh nắng, gió, lạnh)",
            "Sử dụng kem chống nắng SPF 30+ hàng ngày, bôi lại thường xuyên",
            "Rửa mặt với nước ấm (không nóng), tránh nước quá lạnh",
            "Tránh các sản phẩm chứa cồn và hương liệu, chọn sản phẩm dịu nhẹ",
            "Quản lý căng thẳng và ngủ đủ giấc (7-8 giờ/đêm)",
            "Tránh tắm nước nóng và phòng xông hơi",
            "Sử dụng sữa rửa mặt dịu nhẹ, không chứa xà phòng",
            "Dưỡng ẩm da nhẹ nhàng, tránh chà xát mạnh",
            "Tránh các hoạt động gây đổ mồ hôi quá nhiều",
            "Mặc quần áo rộng rãi, thoáng mát",
            "Tránh các sản phẩm chứa retinoid hoặc axit mạnh",
            "Khám bác sĩ da liễu để được tư vấn điều trị phù hợp"
        ],
        "diet": [
            "Tránh thức ăn cay nóng và nhiều gia vị",
            "Hạn chế rượu bia và đồ uống nóng (cà phê, trà nóng)",
            "Tránh thực phẩm chế biến sẵn và đồ ăn nhanh",
            "Tăng cường thực phẩm chống viêm (cá hồi, rau xanh, quả mọng)",
            "Uống đủ nước ở nhiệt độ phòng (2-3 lít/ngày)",
            "Hạn chế thực phẩm có chỉ số đường huyết cao",
            "Tăng cường omega-3 từ cá và các loại hạt",
            "Bổ sung probiotic để cân bằng hệ vi sinh",
            "Tránh thực phẩm chứa nhiều histamine (phô mai, thịt xông khói)",
            "Tăng cường thực phẩm giàu chất chống oxy hóa",
            "Bổ sung vitamin C và E từ thực phẩm tự nhiên",
            "Uống trà xanh hoặc trà hoa cúc để làm dịu"
        ]
    },
    "milia": {
        "name_vi": "Mụn thịt (Milia)",
        "description": "Các nang nhỏ chứa keratin dưới da",
        "lifestyle": [
            "Tẩy tế bào chết nhẹ nhàng 1-2 lần/tuần với sản phẩm dịu nhẹ",
            "Dưỡng ẩm da đầy đủ, đặc biệt sau khi tắm",
            "Tránh nặn hoặc cố gắng loại bỏ tại nhà để tránh nhiễm trùng",
            "Sử dụng sản phẩm chăm sóc da nhẹ nhàng, không chứa hương liệu",
            "Bảo vệ da khỏi ánh nắng bằng kem chống nắng SPF 30+",
            "Rửa mặt 2 lần/ngày với sữa rửa mặt dịu nhẹ",
            "Tẩy trang sạch sẽ trước khi ngủ",
            "Tránh sử dụng sản phẩm quá dầu hoặc quá khô",
            "Giữ da sạch sẽ nhưng không rửa quá nhiều",
            "Sử dụng máy tạo độ ẩm trong phòng ngủ",
            "Tránh stress và ngủ đủ giấc",
            "Tham khảo bác sĩ da liễu nếu muốn điều trị chuyên sâu"
        ],
        "diet": [
            "Chế độ ăn cân bằng với đầy đủ chất dinh dưỡng",
            "Uống đủ nước (2-3 lít/ngày) để giữ ẩm da",
            "Tăng cường vitamin A (cà rốt, khoai lang, rau xanh đậm) và E (các loại hạt)",
            "Hạn chế thực phẩm chế biến sẵn và đồ ăn nhanh",
            "Ăn nhiều rau củ và trái cây giàu chất chống oxy hóa",
            "Tăng cường omega-3 từ cá và các loại hạt",
            "Bổ sung kẽm từ thực phẩm (hàu, thịt đỏ, các loại hạt)",
            "Hạn chế đường và thực phẩm có chỉ số đường huyết cao",
            "Tăng cường thực phẩm giàu biotin (trứng, các loại hạt)",
            "Bổ sung vitamin C từ cam, ổi, ớt chuông",
            "Uống trà xanh để chống oxy hóa",
            "Tăng cường thực phẩm giàu collagen (nước hầm xương, cá)"
        ]
    },
    "none": {
        "name_vi": "Không phát hiện vấn đề",
        "description": "Da có vẻ khỏe mạnh",
        "lifestyle": [
            "Tiếp tục chăm sóc da hàng ngày",
            "Sử dụng kem chống nắng",
            "Dưỡng ẩm da đầy đủ",
            "Tẩy trang sạch sẽ trước khi ngủ",
            "Khám da định kỳ"
        ],
        "diet": [
            "Duy trì chế độ ăn cân bằng",
            "Uống đủ nước",
            "Tăng cường trái cây và rau củ",
            "Hạn chế thực phẩm chế biến sẵn",
            "Bổ sung đầy đủ vitamin và khoáng chất"
        ]
    }
}

# Ngưỡng confidence để hiển thị cảnh báo
HIGH_CONFIDENCE_THRESHOLD = 0.7
MEDIUM_CONFIDENCE_THRESHOLD = 0.5


def generate_health_issue_info(results, detection_confidences):
    """
    Tạo thông tin về vấn đề sức khỏe dựa trên kết quả classification và detection
    Bao gồm TẤT CẢ các bệnh da liễu được phân loại và TẤT CẢ vấn đề thẩm mỹ được phát hiện
    
    Args:
        results: List các kết quả detection và classification
        detection_confidences: List các confidence từ object detection
    
    Returns:
        str: Text mô tả đầy đủ tất cả vấn đề sức khỏe hoặc None nếu confidence thấp
    """
    if not results:
        return None
    
    # Thu thập TẤT CẢ các vấn đề được phát hiện
    diseases = []  # (class_name, combined_confidence, detection_conf, description)
    cosmetics = []  # (detected_class, detection_conf, description)
    
    for i, result in enumerate(results):
        detected_class = result.get('detected_class', '')
        detection_conf = detection_confidences[i] if i < len(detection_confidences) else 0
        requires_classification = result.get('requires_classification', False)
        
        if requires_classification:
            # Xử lý bệnh da liễu có classification
            disease_pred = result.get('disease_prediction', {})
            if disease_pred:
                class_name = disease_pred.get('class_name', 'none')
                classification_conf = disease_pred.get('confidence', 0)
                
                # Tính confidence tổng hợp (trung bình có trọng số)
                combined_confidence = (classification_conf * 0.6 + detection_conf * 0.4)
                
                # Thêm tất cả bệnh có confidence >= MEDIUM_CONFIDENCE_THRESHOLD và khác 'none'
                if combined_confidence >= MEDIUM_CONFIDENCE_THRESHOLD and class_name != 'none':
                    disease_info = DISEASE_INFO.get(class_name, DISEASE_INFO['none'])
                    disease_name_vi = disease_info['name_vi']
                    description = disease_info['description']
                    diseases.append((class_name, combined_confidence, detection_conf, disease_name_vi, description))
        else:
            # Xử lý vấn đề thẩm mỹ/thông thường (chỉ có detection)
            if detection_conf >= MEDIUM_CONFIDENCE_THRESHOLD:
                cosmetic_info = COSMETIC_ISSUES.get(detected_class)
                if cosmetic_info:
                    issue_name_vi = cosmetic_info['name_vi']
                    description = cosmetic_info['description']
                    cosmetics.append((detected_class, detection_conf, issue_name_vi, description))
    
    # Tạo text mô tả - hiển thị TẤT CẢ các vấn đề
    text_parts = []
    
    # Sắp xếp theo confidence
    diseases.sort(key=lambda x: x[1], reverse=True)
    cosmetics.sort(key=lambda x: x[1], reverse=True)
    
    # Trường hợp 1: Có bệnh da liễu - hiển thị TẤT CẢ các bệnh được phát hiện
    if diseases:
        # Thêm bệnh da liễu chính (confidence cao nhất)
        first_disease = diseases[0]
        disease_name_vi = first_disease[3]
        description = first_disease[4]
        confidence = first_disease[1]
        
        text_parts.append(f"Bạn có thể đang gặp vấn đề về: {disease_name_vi} ({confidence*100:.1f}%). {description}")
        
        # Thêm tất cả các bệnh da liễu khác được phát hiện
        for disease in diseases[1:]:
            disease_name_vi = disease[3]
            description = disease[4]
            confidence = disease[1]
            
            text_parts.append(f"Ngoài ra, cũng phát hiện: {disease_name_vi} ({confidence*100:.1f}%). {description}")
        
        # Nếu có vấn đề thẩm mỹ kèm theo, thêm vào
        if cosmetics:
            for cosmetic in cosmetics:
                issue_name_vi = cosmetic[2]
                description = cosmetic[3]
                confidence = cosmetic[1]
                
                text_parts.append(f"Thêm vào đó, phát hiện vấn đề thẩm mỹ: {issue_name_vi} ({confidence*100:.1f}%). {description}")
    
    # Trường hợp 2: Chỉ có vấn đề thẩm mỹ (không có bệnh da liễu)
    elif cosmetics:
        # Vấn đề thẩm mỹ đầu tiên
        first_cosmetic = cosmetics[0]
        issue_name_vi = first_cosmetic[2]
        description = first_cosmetic[3]
        confidence = first_cosmetic[1]
        
        text_parts.append(f"Phát hiện vấn đề về: {issue_name_vi} ({confidence*100:.1f}%). {description}")
        
        # Tất cả các vấn đề thẩm mỹ khác
        for cosmetic in cosmetics[1:]:
            issue_name_vi = cosmetic[2]
            description = cosmetic[3]
            confidence = cosmetic[1]
            
            text_parts.append(f"Ngoài ra, phát hiện vấn đề: {issue_name_vi} ({confidence*100:.1f}%). {description}")
    
    if not text_parts:
        return None
    
    return " ".join(text_parts) + "."


def generate_lifestyle_suggestions(results, detection_confidences):
    """
    Tạo gợi ý về lối sống và ăn uống dựa trên TẤT CẢ các vấn đề được phát hiện
    Kết hợp gợi ý từ cả tất cả bệnh da liễu và tất cả vấn đề thẩm mỹ
    
    Args:
        results: List các kết quả detection và classification
        detection_confidences: List các confidence từ object detection
    
    Returns:
        dict: Dictionary chứa lifestyle và diet suggestions từ tất cả vấn đề
    """
    if not results:
        return {
            "lifestyle": DISEASE_INFO['none']['lifestyle'],
            "diet": DISEASE_INFO['none']['diet']
        }
    
    # Thu thập TẤT CẢ các vấn đề được phát hiện
    diseases = []  # (class_name, confidence)
    cosmetics = []  # (detected_class, confidence)
    
    for i, result in enumerate(results):
        detected_class = result.get('detected_class', '')
        detection_conf = detection_confidences[i] if i < len(detection_confidences) else 0
        requires_classification = result.get('requires_classification', False)
        
        if requires_classification:
            # Xử lý bệnh da liễu có classification
            disease_pred = result.get('disease_prediction', {})
            if disease_pred:
                class_name = disease_pred.get('class_name', 'none')
                classification_conf = disease_pred.get('confidence', 0)
                combined_confidence = (classification_conf * 0.6 + detection_conf * 0.4)
                
                # Thêm TẤT CẢ bệnh có confidence >= MEDIUM_CONFIDENCE_THRESHOLD
                if combined_confidence >= MEDIUM_CONFIDENCE_THRESHOLD and class_name != 'none':
                    diseases.append((class_name, combined_confidence))
        else:
            # Xử lý vấn đề thẩm mỹ/thông thường
            if detection_conf >= MEDIUM_CONFIDENCE_THRESHOLD:
                cosmetic_info = COSMETIC_ISSUES.get(detected_class)
                if cosmetic_info:
                    cosmetics.append((detected_class, detection_conf))
    
    # Kết hợp gợi ý từ TẤT CẢ các vấn đề
    lifestyle = []
    diet = []
    
    if diseases:
        # Sắp xếp theo confidence
        diseases.sort(key=lambda x: x[1], reverse=True)
        
        # Thêm gợi ý từ TẤT CẢ các bệnh da liễu
        for disease_name, confidence in diseases:
            disease_info = DISEASE_INFO.get(disease_name, DISEASE_INFO['none'])
            
            # Thêm tất cả gợi ý từ mỗi bệnh (tránh trùng lặp)
            for suggestion in disease_info['lifestyle']:
                if suggestion not in lifestyle:
                    lifestyle.append(suggestion)
            
            for suggestion in disease_info['diet']:
                if suggestion not in diet:
                    diet.append(suggestion)
        
        # Thêm gợi ý chung
        if "Nên tham khảo ý kiến bác sĩ da liễu để có chẩn đoán chính xác" not in lifestyle:
            lifestyle.append("Nên tham khảo ý kiến bác sĩ da liễu để có chẩn đoán chính xác")
    
    # Thêm gợi ý từ TẤT CẢ vấn đề thẩm mỹ
    if cosmetics:
        cosmetics.sort(key=lambda x: x[1], reverse=True)
        
        for detected_class, confidence in cosmetics:
            cosmetic_info = COSMETIC_ISSUES.get(detected_class)
            
            if cosmetic_info:
                # Thêm tất cả gợi ý từ mỗi vấn đề thẩm mỹ (tránh trùng lặp)
                for suggestion in cosmetic_info['lifestyle']:
                    if suggestion not in lifestyle:
                        lifestyle.append(suggestion)
                
                for suggestion in cosmetic_info['diet']:
                    if suggestion not in diet:
                        diet.append(suggestion)
    
    # Thêm gợi ý chung cuối cùng
    if not diet:
        diet.append("Duy trì chế độ ăn cân bằng và lành mạnh")
    if not lifestyle:
        lifestyle.append("Tiếp tục chăm sóc da hàng ngày")
    
    # Nếu không có vấn đề nào, dùng gợi ý chung
    if not diseases and not cosmetics:
        return {
            "lifestyle": DISEASE_INFO['none']['lifestyle'],
            "diet": DISEASE_INFO['none']['diet']
        }
    
    return {
        "lifestyle": lifestyle,
        "diet": diet
    }

