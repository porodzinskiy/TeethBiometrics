#Импорт необходимых библиотек
import cv2
import os
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
#Импорт моделей нейросетей
faces = cv2.CascadeClassifier('frontalface_default.xml') #Модель для поиска лиц
mouth = cv2.CascadeClassifier('mcs_mouth.xml') #Модель для поиска рта

#Регулируемые параметры
key = b'ABCDabcdABCDabcd' #Ключ для шифрования и дешифрования (16 байт)
webcam_input = 0 #Выбор используемой вебкамеры (None- для использования галереи)
image_path = 'images/user.jpg' #Путь к изображению, подаваемому на вход считывателя
users_dir = 'users'  #Путь к базе данных
image_error = cv2.imread('images/not_found.jpg') #Изображение, сообщаещее об ошибке
find_mouth_flag = True #Флаг использования нейронной сети для нахождения рабочей области изображения
bilateral_filter_flag = True #Флаг использования Билатерального фильтра вместо фильтра Гаусса
debug_save_flag = False #Флаг для отладки
debug_save_path = 'images/debug' #Путь для сохранения файлов отладки

#Функция, выполняющая роль считывателя
def reader():
    if webcam_input != None: #Если захват из веб камеры
        success, image = capture.read() #Захват изображения
        if success: return image #Возврат изображения, если захват успешен
    else:
        if os.path.exists(image_path): #Проверка на наличие файла
            image = cv2.imread(image_path)  #Извлечение изображения
            return image
    print('Неверный захват изображения') #Сообщение в консоль об ошибке
    quit()

#Функция, выполняющая роль экстрактора свойств
def extractor(image):
    if find_mouth_flag: image_squares = image #Создание копии изображения для рисования квадратов

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #Перевод изображения в оттенки серого
    if debug_save_flag: cv2.imwrite(f'images/results/{debug_save_path}/gray.jpg', image) #Отладка

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  #Адаптивное выравнивание гистограммы (по блокам)
    image = clahe.apply(image) #Применение выравнивания гистограммы
    if debug_save_flag: cv2.imwrite(f'images/results/{debug_save_path}/hist.jpg', image)  #Отладка

    if find_mouth_flag: #Если применяется нейронная сеть для обнаружения рабочей области
        faces_pos = faces.detectMultiScale(image, scaleFactor=1.1, minNeighbors=4)  #Поиск лиц
        if len(faces_pos) != 0: #Если присутствуют лица на фото
            for (x_faces, y_faces, w_faces, h_faces) in faces_pos:  #Для всех найденных лиц
                cv2.rectangle(image_squares, (x_faces, y_faces), (x_faces + w_faces, y_faces + h_faces), (0, 0, 255),
                              thickness=2)  #Рисование квадратов для наглядности
                h_del = int(h_faces // 2)
                image = image[(y_faces + h_del):(y_faces + h_faces),
                        x_faces:(x_faces + w_faces)]  #Обрезка изображения до области поиска рта (нижняя половина лица)
                mouth_pos = mouth.detectMultiScale(image, scaleFactor=1.1, minNeighbors=100)  #Поиск рта
                if len(mouth_pos) == 1: #Если один рот на лице
                    for (x_mouth, y_mouth, w_mouth, h_mouth) in mouth_pos: #Выделение координатов рта
                        cv2.rectangle(image_squares, (x_mouth + x_faces, y_mouth + y_faces + h_del),
                                  (x_mouth + x_faces + w_mouth, y_mouth + y_faces + h_del + h_mouth), (0, 255, 0),
                                  thickness=2)  #Рисование квадратов для наглядности
                        image = image[y_mouth:(y_mouth + h_mouth),
                                      x_mouth:(x_mouth + w_mouth)]  #Обрезка изображения до области рта
                    break
                else: image = image_error #Применение изображения ошибки в случае отсутствии рта
        else: image = image_error  # Применение изображения ошибки в случае отсутствии рта
        if debug_save_flag: cv2.imwrite(f'images/results/{debug_save_path}/squares.jpg', image_squares) #Отладка
        cv2.imshow('squares', image_squares)  #Демонстрация изображения с найденными лицами и ртом

    image = cv2.resize(image, (352, 288))  #Изменение размера изображения
    if debug_save_flag: cv2.imwrite(f'images/results/{debug_save_path}/mouth.jpg', image)  #Отладка
    cv2.imshow('mouth', image)  #Демонстрация изображения рабочей области

    if not bilateral_filter_flag: image = cv2.GaussianBlur(image, (7, 7), 1)  #Размытие изображения по Гауссу
    else: image = cv2.bilateralFilter(image, 9, 75, 75)  #Размытие изображения с применением фильтра
    if debug_save_flag: cv2.imwrite(f'images/results/{debug_save_path}/blur.jpg', image)  #Отладка

    image = cv2.Canny(image, 40, 90)  #Выделение контуров методом детектера краев Canny
    if debug_save_flag: cv2.imwrite(f'images/results/{debug_save_path}/edge.jpg', image)  #Отладка
    cv2.imshow('edges', image) #Демонстрация изображения контуров
    return image

#Функции, выполняющие роль устройства сопоставления
def comparison_sift(image): #Функция, извлекающая ключевые точки и дескрипторы
    sift = cv2.SIFT_create()  #Создание экстрактора SIFT
    keypoints, descriptors = sift.detectAndCompute(image, None)  #Выделение ключевых точек и дескрипторов
    return keypoints, descriptors
def comparison_fbn(keypoints1, descriptors1, keypoints2, descriptors2): #Функция, производящая
                                    #сопоставление двух фотографий методом сопоставления FlannBasedMatcher
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED) #Сопоставление сопоставителя FBN
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2) #Расчет расстояния между дескрипторами
    good_matches = [] #Создание массива для записи хороших совпадений
    for m, n in matches: #Проверка всех совпадений
        if m.distance < 0.75 * n.distance: #Если рассточние выше порогового значения
            good_matches.append(m) #Запись хорошего совпадения
    similarity_ratio = len(good_matches) / max(len(keypoints1),
                                               len(keypoints2))  #Отношение хороших сопоставлений ко всем
    return similarity_ratio

#Функции, выполняющие роль биометрической базы данных
def database_save(image): #Функция для сохранения параметров пользователя
    name = input('Введите имя пользователя (на английском): ')  #Выбор имени пользователя
    image_bytes = cv2.imencode('.jpg', image)[1].tobytes() #Байтовое представление изображения
    encrypted_image_bytes = cipher.encrypt(pad(image_bytes, AES.block_size)) #Шифрование
    with open(f'{users_dir}/{name}.txt', 'wb') as f: #Открытие файла
        f.write(encrypted_image_bytes) #Запись в файл
    print(f'Сохранен пользователь {name}')  #Вывод в консоль сообщения об успешной записи
def database_load(): #Функция для загрузки параметров пользователей
    files = os.listdir(users_dir)  #Получение списка файлов из директории пользователей
    names = []  # Создание списка имен пользователей
    keypoints_list = []  #Создание списка ключевых точек пользователей
    descriptors_list = [] #Создание списка дескрипторов пользователей
    for file in files:  # Запись в массив всех пользователей
        with open(f'{users_dir}/{file}', 'rb') as f: #Открытие файла
            encrypted_image_bytes = f.read() #Прочтение файла
        decrypted_image_bytes = unpad(cipher.decrypt(encrypted_image_bytes), AES.block_size) #Расшифрование
        decrypted_image_array = np.frombuffer(decrypted_image_bytes, dtype=np.uint8) #Создание массива
        decrypted_image = cv2.imdecode(decrypted_image_array, cv2.IMREAD_COLOR) #Декодирование и перевод в изображение
        keypoints, descriptors = comparison_sift(decrypted_image) #Получение изображения,
                                                                #получение ключевых точек и дескрипторов
        names.append(file[:-4]) #Запись в список имен пользователей
        keypoints_list.append(keypoints) #Запись в список ключевых точек пользователей
        descriptors_list.append(descriptors) #Запись в список дескрипторов пользователей
    return names, keypoints_list, descriptors_list

#При старте программы
if webcam_input != None: #Если захват происходит из веб камеры
    capture = cv2.VideoCapture(webcam_input, cv2.CAP_DSHOW)  #Выбор веб камеры
    capture.set(3, 1280) #Приведение к необходимому размеру по ширине
    capture.set(4, 720) #Приведение к необходимому размеру по высоте
cipher = AES.new(key, AES.MODE_ECB) #Инициализация класса
names, keypoints_list, descriptors_list = database_load() #Получение из БД всех имен, ключевых точек и дескрипторов


while True:
    image = extractor(reader()) #Получения изображения на выходе экстрактора
    if cv2.waitKey(1) & 0xFF == ord('w'): #Ожидание ключа для записи пользователя
        database_save(image) #Сохранение в базу данных
        quit() #Завершение работы программы
    keypoints, descriptors = comparison_sift(image)  #Получение ключевых точек и дескриптора изображения
                                                            #из входного потока после экстракции свойств
    for i in range(len(names)): #Сравнение со всеми пользователями
        if comparison_fbn(keypoints, descriptors, keypoints_list[i], descriptors_list[i]) > 0.1: #Выше порогового значения
            print(f'Идентификация-аутентификация успешна. Пользователь {names[i]}') #Сообщение в консоль
            quit() #Выход из программы
    print('Идентификация неуспешна') #Сообщение в консоль
    if cv2.waitKey(1) & 0xFF == ord('q'): #Ожидание ключа для выхода из программы
        quit() #Выход из программы