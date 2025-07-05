import cv2
import numpy as np
import glob
import os

# initial values (не используются в текущей обработке)
gamma_value = 1
contrast = 1.3
saturation = 1.2
brightness = 1.1


def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)


# Найти все RAW-файлы в директории
files = glob.glob("/home/danil/photos/*.raw")
files.sort()

if len(files) > 0:
    for file_path in files:
        # Открыть RAW-файл
        with open(file_path, 'rb') as f:
            image = np.fromfile(f, dtype=np.uint8)

        # Определение размера изображения
        cols, rows, valid = None, None, 0
        size_mappings = {
            1658880: (1536, 864, 1),  # Pi3
            14929920: (4608, 2592, 1),
            3732480: (2304, 1296, 1),
            384000: (640, 480, 1),
            2562560: (1664, 1232, 1),
            10171392: (3280, 2464, 1),
            2592000: (1920, 1080, 1),
            1586304: (1296, 972, 1),
            4669440: (2048, 1520, 2),  # PiHQ
            3317760: (2048, 1080, 2),
            18580480: (4056, 3040, 2)
        }

        if image.size in size_mappings:
            cols, rows, valid = size_mappings[image.size]
        else:
            print(f"Unsupported file size {image.size} for {file_path}")
            continue

        # Предобработка для специальных случаев
        if image.size == 10171392:
            image = image.reshape(-1, 4128)
            image = np.delete(image, np.s_[4099:4128], axis=1)
        elif image.size == 18580480:
            image = image.reshape(-1, 6112)
            image = np.delete(image, np.s_[6083:6112], axis=1)
        elif image.size == 1586304:
            image = image.reshape(-1, 1632)
            image = np.delete(image, np.s_[1619:1632], axis=1)

        # Извлечение данных изображения
        if valid == 1:
            A = image.reshape(-1, 5)
            A = np.split(A, [4, 5], axis=1)[0]
        else:
            A = image.reshape(-1, 3)
            A = np.split(A, [2, 3], axis=1)[0]

        F = A.reshape(rows, cols)
        #F = cv2.resize(F, dsize=(int(cols / 4), int(rows / 4)), interpolation=cv2.INTER_CUBIC)

        # Генерация имени файла
        base_name = os.path.basename(file_path)
        tiff_name = os.path.splitext(base_name)[0] + '.tiff'
        output_path = os.path.join('/home/danil/photos/processed', tiff_name)  # Измените путь при необходимости

        # Создание директории если нужно
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Сохранение в TIFF
        cv2.imwrite(output_path, F)
        print(f"Saved: {output_path}")

    print("All files processed!")
else:
    print("No RAW files found in the specified directory")