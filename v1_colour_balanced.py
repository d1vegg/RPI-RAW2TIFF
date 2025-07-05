import cv2
import numpy as np
import glob
import os

# ===== ПАРАМЕТРЫ ОБРАБОТКИ (регулируйте по необходимости) =====
gamma_value = 1.8  # Гамма-коррекция (1.8-2.2)
contrast = 0.3  # Контрастность CLAHE (0.3-0.5)
wb_manual = [1.2, 1.0, 1.3]  # Баланс белого [R, G, B]
black_level = 400  # Уровень чёрного (300-500 для 10-bit)
shadows_boost = 0.25  # Усиление теней (0.2-0.4)
highlights_compression = 0.3  # Сжатие светов (0.2-0.5)


# =============================================================

def unpack_10bit_packed(data):
    """Распаковка 10-битных данных в 16-битный массив"""
    data = data.astype(np.uint16).reshape(-1, 5)
    lsb = data[:, 4]

    pixels = np.empty((data.shape[0], 4), dtype=np.uint16)
    pixels[:, 0] = (data[:, 0] << 2) | (lsb & 0x03)
    pixels[:, 1] = (data[:, 1] << 2) | ((lsb >> 2) & 0x03)
    pixels[:, 2] = (data[:, 2] << 2) | ((lsb >> 4) & 0x03)
    pixels[:, 3] = (data[:, 3] << 2) | ((lsb >> 6) & 0x03)

    return pixels.flatten()


def apply_black_level_correction(bayer, black_level):
    """Коррекция уровня чёрного с поднятием теней"""
    corrected = np.clip(bayer.astype(np.int32) - black_level, 0, 65535)

    # Нелинейное поднятие теней
    shadows_mask = (corrected < 4096) & (corrected > 0)
    corrected[shadows_mask] = corrected[shadows_mask] * (1 + shadows_boost * (1 - corrected[shadows_mask] / 4096))

    return np.clip(corrected, 0, 65535).astype(np.uint16)


def gamma_correction(src, gamma=1.8):
    """Гамма-коррекция с защитой светов"""
    # Создаем маску для защиты ярких областей
    _, light_mask = cv2.threshold(src, 220, 255, cv2.THRESH_BINARY)

    # Применяем гамму
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    corrected = cv2.LUT(src, table)

    # Восстанавливаем оригинальные света
    corrected = np.where(light_mask > 0, src, corrected)
    return corrected


def enhance_tonal_range(rgb_image):
    """Оптимизация тонального диапазона"""
    # Рассчитываем процентили (игнорируя чистый черный)
    low_val = np.percentile(rgb_image[rgb_image > 100], 2)
    high_val = np.percentile(rgb_image, 98)

    # Растягиваем гистограмму
    scale = 65535.0 / (high_val - low_val + 1e-7)
    rgb_image = (rgb_image - low_val) * scale

    # Компрессия светов (предотвращение пересветов)
    overexposed = rgb_image > 58000
    rgb_image[overexposed] = 58000 + (rgb_image[overexposed] - 58000) * highlights_compression

    return np.clip(rgb_image, 0, 65535).astype(np.uint16)


# Обработка файлов
home_dir = os.path.expanduser("~")
files = glob.glob(os.path.join(home_dir, "photos/*.raw"))
files.sort()

for file_path in files:
    print(f"Обработка: {file_path}")

    # Чтение RAW-файла
    with open(file_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.uint8)

    # Параметры изображения (для Raspberry Pi HQ Camera)
    cols, rows = 4608, 2592
    bayer_pattern = cv2.COLOR_BAYER_RGGB2BGR

    # Проверка формата
    if raw_data.size != 14929920:  # 10-bit
        print(f"Неподдерживаемый размер файла: {raw_data.size}")
        continue

    try:
        # Распаковка 10-битных данных
        unpacked = unpack_10bit_packed(raw_data)
        bayer_matrix = unpacked.reshape(rows, cols)
        bayer_matrix = (bayer_matrix << 6).astype(np.uint16)  # Масштабирование до 16 бит

        # Коррекция уровня чёрного
        bayer_corrected = apply_black_level_correction(bayer_matrix, black_level)

        # Демозаикинг
        rgb_image = cv2.cvtColor(bayer_corrected, bayer_pattern)

        # Применение баланса белого
        rgb_image = rgb_image.astype(np.float32)
        rgb_image[..., 0] *= wb_manual[0]  # Красный канал
        rgb_image[..., 1] *= wb_manual[1]  # Зелёный канал
        rgb_image[..., 2] *= wb_manual[2]  # Синий канал

        # Оптимизация тонального диапазона
        rgb_16bit = enhance_tonal_range(rgb_image)

        # Конвертация в 8-бит
        rgb_8bit = (rgb_16bit // 256).astype(np.uint8)

        # Мягкое шумоподавление
        rgb_8bit = cv2.bilateralFilter(rgb_8bit, 5, 25, 25)

        # Коррекция контраста в LAB пространстве
        lab = cv2.cvtColor(rgb_8bit, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # CLAHE только для средних тонов
        midtones_mask = cv2.inRange(l_channel, 30, 220)
        clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(12, 12))
        enhanced_l = np.where(midtones_mask, clahe.apply(l_channel), l_channel)

        # Собираем обратно LAB изображение
        lab = cv2.merge([enhanced_l, a_channel, b_channel])
        rgb_8bit = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Гамма-коррекция
        rgb_8bit = gamma_correction(rgb_8bit, gamma_value)

        # Финальное повышение насыщенности
        hsv = cv2.cvtColor(rgb_8bit, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.1, 0, 255).astype(np.uint8)
        rgb_8bit = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Сохранение результата
        output_path = file_path.replace('.raw', '.tif')
        cv2.imwrite(output_path, rgb_8bit)
        print(f"Сохранено: {output_path}")

    except Exception as e:
        print(f"Ошибка обработки: {str(e)}")
        continue

print("Обработка всех файлов завершена!")