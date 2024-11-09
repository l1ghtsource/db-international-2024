# db_international_2024
 
 Как запустить трейн скрипты

 1) Если репо ещё не склонирован, делаем 
 ```
 git clone https://github.com/l1ghtsource/db_international_2024.git
 cd db_international_2024
 ```
 2) Создаём или используем готовое виртуальное окружение 
 ```
 conda create --name hack
 conda activate hack
 (АААААААААААА КОНДА ПОТОМ ПОПРАВИТЕ)
```
Или
 ```
 source environments/hack/bin/activate
 ```
3) Устанавливаем необходиимые зависимости
```
pip install -r requirements.txt
```
4) Допишу после мерджа 

```
python /src/main.py --mode train --data-path <path_to_your_dataset> 
            --save-model-path ./logs/<name_of_your_experiment>.pth --wand-key <your_wandb_api_key>
``` 