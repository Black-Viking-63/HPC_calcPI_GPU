# HPC_вычисление_числа_PI_методом_Монте-Карло
В данном репозитории расположено решение лабораторной работы №2 Вычисление числа PI.<br><br>
# Задание на лабораторную и аппаратная база
Задача: реализовать алгоритм вычисления числа PI.<br>
Учитывая количество точек N, сгенерируйте случайное распределение в области (0; 0)~(1; 1) и вычислите число используя CPU и GPU.
Полученные значения должны быть распечатаны вместе со временем выполнения.<br>
_Входные данные: количество точек.<br>
Выходные данные: время выполнения и полученные числа PI._<br>
Реализация должна содержать 2 функции вычисления числа PI: на CPU и на GPU с применением CUDA.<br><br>
Для реализации данной задачи использовалась следующая аппаратная база:<br>
Центральный процессор: _Intel Xeon E5-2620 v3 @ 2,4 GHz._<br>
Оперативная память: _Kllisre DDR4, 2 × 8 GB, 1600 MHz, DualChannel._<br>
Графический процессор: _PALIT GTX 1650 SUPER, 4GB VRAM GDDR6._<br><br>
# Реализация алгоритма
Рассмотрим более подробно реализацию данного алгоритма. <br>
Вычисление числа PI осуществляется с использованием вероятностного метода Монте-Карло. Алгоритм данного метода представлен на рисунке 1.<br><br>
![Screenshot](screenshot.png)<br><br>
Фнукция _gpu_calculation_pi()_ производит вычисление числа Pi на графическом процессоре. Функция _cpu_calculation_pi()_ производит вычисления на центральном процессоре.<br>
Генерация данных производилась на графическом процессоре при помощи библиотеки CURAND, в которой существуютя для этого функции. Функция curand_init инициализирует генератор случайных чисел CUDA. Функция curand_uniform возвращает последовательность псевдослучайных чисел с плавающей запятой, равномерно распределенных между 0,0 и 1,0.<br><br>
# Результаты работы программы
Теперь рассмотрим результаты работы программы.
