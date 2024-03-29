import os
import requests
import time


# 1982 ~ 2019年までの北西太平洋月平均海面水温のヒートマップ画像を、
# 台風がよく発生する、6 ~ 10月について取得する.


def download_img(url, file_name):
    print('downloading:', url)
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(r.content)


def main():
    is_anom = True
    year = 1982
    end_year = 2019
    months = ['06', '07', '08', '09', '10']
    img_dir = '../images'
    url_template = 'https://www.data.jma.go.jp/gmd/kaiyou/data/db/kaikyo/monthly/image/HT/{year}'

    if is_anom:
        url_template = os.path.join(url_template, 'sstM_anom_HT{year}{month}.png')
        img_dir = os.path.join(img_dir, 'anom')
        os.makedirs(img_dir, exist_ok=True)
    else:
        url_template = os.path.join(url_template, 'sstM_HT{year}{month}.png')

    while year <= end_year:
        for month in months:
            filename = '{}_{}.png'.format(year, month)
            download_img(url_template.format(year=year, month=month),
                         os.path.join(img_dir, filename))
            time.sleep(0.1)
        year += 1


if __name__ == "__main__":
    main()
