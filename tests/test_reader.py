from app.reader import MeterReader

def test_normalize_common_ocr_confusions():
    assert MeterReader.normalize('O12S8') == '01258'

def test_decimal_placement():
    assert MeterReader.normalize('00123456', 3) == '00123.456'
