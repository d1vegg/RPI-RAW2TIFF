import cv2
import numpy as np
import glob
import os

# Параметры обработки
gamma_value = 6
contrast = 0.5
wb_manual = [1.9, 1.0, 1.4]  # Ручные коэффициенты баланса белого [R, G, B]


def unpack_10bit_packed(data):
    """Оптимизированная распаковка 10-битных данных"""
    data = data.astype(np.uint16).reshape(-1, 5)
    lsb = data[:, 4]

    pixels = np.empty((data.shape[0], 4), dtype=np.uint16)
    pixels[:, 0] = (data[:, 0] << 2) | (lsb & 0x03)
    pixels[:, 1] = (data[:, 1] << 2) | ((lsb >> 2) & 0x03)
    pixels[:, 2] = (data[:, 2] << 2) | ((lsb >> 4) & 0x03)
    pixels[:, 3] = (data[:, 3] << 2) | ((lsb >> 6) & 0x03)

    return pixels.flatten()


def apply_black_level_correction(bayer, black_level=256):
    """Коррекция уровня чёрного"""
    return np.clip(bayer.astype(np.int32) - black_level, 0, 65535).astype(np.uint16)


def gamma_correction(src, gamma = 2.2):
    """Гамма-коррекция с использованием LUT"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(src, table)


def calculate_white_balance(rgb_image):
    """Автоматический расчёт баланса белого по серой карте"""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, np.percentile(gray, 70), np.percentile(gray, 99))
    y, x = np.where(mask)
    return np.mean(rgb_image[y, x], axis=0)

# Обработка файлов
home_dir = os.path.expanduser("~")
files = glob.glob(os.path.join(home_dir, "photos/*.raw"))
files.sort()

for file_path in files:
    print(f"Processing: {file_path}")

    with open(file_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.uint8)

    # Определение параметров
    cols, rows = 4608, 2592
    bayer_pattern = cv2.COLOR_BAYER_RGGB2BGR  # Основной паттерн для Raspberry Pi


    if raw_data.size == 14929920:  # 10-bit
        unpacked = unpack_10bit_packed(raw_data)
        bayer_matrix = unpacked.reshape(rows, cols)
        bayer_matrix = (bayer_matrix << 6).astype(np.uint16)  # Масштабирование до 16 бит
    else:
        print(f"Unsupported file size: {raw_data.size}")
        continue

    try:
        # Коррекция уровня чёрного
        bayer_corrected = apply_black_level_correction(bayer_matrix)

        # Демозаик
        rgb_image = cv2.cvtColor(bayer_corrected, bayer_pattern)

        # Ручной баланс белого
        rgb_image = rgb_image.astype(np.float32)
        rgb_image[..., 0] *= wb_manual[0]  # Красный канал
        rgb_image[..., 1] *= wb_manual[1]  # Зелёный канал
        rgb_image[..., 2] *= wb_manual[2]  # Синий канал

        # Автоматическая нормализация
        rgb_image = cv2.normalize(rgb_image, None, 0, 65535, cv2.NORM_MINMAX)
        rgb_16bit = np.clip(rgb_image, 0, 65535).astype(np.uint16)

        # Конвертация в 8 бит для дальнейшей обработки
        rgb_8bit = cv2.normalize(rgb_16bit, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Дополнительные коррекции
        lab = cv2.cvtColor(rgb_8bit, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8))
        lab = cv2.merge([clahe.apply(l), a, b])
        rgb_8bit = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        rgb_8bit = gamma_correction(rgb_8bit, gamma_value)

        # Сохранение
        output_path = file_path.replace('.raw', '.tif')
        cv2.imwrite(output_path, rgb_8bit)
        print(f"Successfully saved: {output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        continue

print("All files processed!")