# Распознавание рукописных цифр на Python + GUI
Приложение для распознавания написанных от руки цифр с помощью набора данных MNIST. Использованы специальные типы глубоких нейронной сетей, которые называются сверточной и рекурентной нейронной сетью. А так же имеется графический интерфейс, в котором можно рисовать цифру и тут же ее узнавать.


### Установка зависимостей
Этот пакет сопровождается с файлом requirments.txt, в котором хранятся основные зависимости. 
Необходимо установить его командой:
```
$cd musical-octo-goggles
$pip install -r requirments.txt
```
Далее запустить файл gui.py.


### Обучение новой модели
Обучение производится с помощью двух скриптов. Первый скрипт ml1.py является рекурентной нейронной сетью с 10 эпохами (датасет проходит через нейронную сеть в прямом и обратном направлении 10 раз). Второй скрипт ml2.py является сверточной и принимает на вход фиксированные размеры и генерирует на выход фиксированный размер. 


### Поддержка
Если у вас возникли сложности или вопросы по использованию пакета, создайте 
[обсуждение][] в данном репозитории или напишите на электронную почту 
<jetigeenn@gmail.com>.

[обсуждение]: https://github.com/AJ-Se7eN/musical-octo-goggles/issues
