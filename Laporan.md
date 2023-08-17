# Laporan Proyek Machine Learning - Aldiansah Permana

## Domain Proyek
Industri perfilman dunia merupakan salah satu industri yang tidak terpengaruh dengan maraknya hiburan digital seperti munculnya media sosial, program televisi yang beragam dan game. Industri film yang terus melakukan produksi ini semakin menambah informasi film yang melimpah di internet. Kondisi ini justru membuat para penikmat film menjadi kebingungan ketika harus  memilih film kesukaannya. Sistem rekomendasi menyediakan  informasi berdasarkan interaksi pengguna dan item yang telah terekam sebelumnya. Penelitian ini akan membahas pembangunan sistem rekomendasi dengan metode pendekatan *Collaborative Filtering*.


## Business Understanding
### Problem Statements
* Bagaimana cara meningkatkan *user experience* saat mencari film yang ingin ditonton?
* Bagaimana cara membuat sistem rekomendasi film menggunakan metode *collaborative filtering*?

### Goals
* Meningkatkan *user experience* saat mencari film yang ingin ditonton.
* Dapat mengimplementasikan metode *collaborative filtering* untuk sistem rekomendasi film.

### Solution statements
Dataset yang digunakan hanya berisi tentang rating atau hasil penilaian pengguna dan genre film, maka solusi yang sangat tepat untuk masalah ini adalah dengan menggunakan *collaborative filtering*.
[Collaborative Filtering](https://medium.com/@ranggaantok/bagaimana-sistem-rekomendasi-berkerja-e749dac64816): *collaborative filtering* adalah suatu konsep dimana opini dari pengguna lain yang ada digunakan untuk memprediksi item yang mungkin disukai/diminati oleh seorang pengguna.
Pada *collaborative filtering attribut* yang digunakan bukan konten tetapi *user behaviour*. contohnya kita merekomendasikan suatu item berdasarkan dari riwayat *rating* dari *user* tersebut maupun *user* lain.
|       | User1 | User2 | User3 | User4 |
|-------|-------|-------|-------|-------|
| Film1 | 5     | 3     |       |       |
| Film2 |       | 4     |       |       |
| Film3 | 3     | 5     |       | 5     |
| Film4 | 3     |       | 5     |       |
| Film5 | 4     |       | 5     |       |



## Data Understanding
Untuk dataset sendiri diambil dari [Movie Lens Dataset](https://www.kaggle.com/aigamer/movie-lens-dataset) yang berada di platform [kaggle](https://www.kaggle.com/). Berikut adalah keterangan mengenai maksud dari variable - variable atau kolom tersebut:

* movies.csv
    * movieId : Unique Id disediakan untuk setiap Film
    * title : Nama film dengan Tahun dalam tanda kurung
    * genres : Genre pada film tersebut
* ratings.csv
    * userId : Unique Id disediakan untuk setiap Pengguna
    * movieId : Unique Id disediakan untuk setiap Film
    * rating : Penilaian pengguna terhadap film terkait
    * timestamp : Kode waktu film
* tags.csv
    * userId : Unique Id disediakan untuk setiap Pengguna
    * movieId : Unique Id disediakan untuk setiap Film
    * tag : Metadata yang dibuat pengguna tentang film. 
    * timestamp : Kode waktu film

Berikut adalah overview dari dataset tersebut setelah dijadikan dataframe:
movie_df adalah isi dari dataset movies.csv.
![movie_df](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/3f5248ad-cd2d-4de2-a889-db82fbc82fff)

rating_df adalah isi dari dataset rating.csv yang sebelumnya telah dihilangkan kolom timestamp nya.
![rating_df](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/b1769750-4a34-4441-88c7-10ff385509ae)

tags_df adalah isi dari dataset tags.csv.
![tag_df](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/78b533e9-c1cd-420a-99b8-c084dc0876e5)

Cek informasi di setiap dataset

![movie_df1](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/2ef8cd55-3b8a-4003-94da-e74ad92d0fbc)

![rating_df1](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/d9a44eef-b0b0-40a2-bf7d-1836bfe07f83)

![tag_df1](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/5dfbfcdf-43a6-43f7-bce2-96f2a8da2787)



Cek data null Data null dapat membuat suatu hasil prediksi model menjadi tidak akurat. Cara untuk melihat apakah data ini mengandung null atau tidak adalah dengan menggunakan method dari library pandas yaitu isnull(). Berikut adalah hasil dari cek data null oleh pandas : 
![movie_df isnull](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/9796f3bd-4741-4b6e-adfc-65a6813afec6)

![rating_df isnull](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/9abbdd6e-529b-42ab-a253-1c85eab73df9)

![tags_df isnull](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/b1053617-a467-4d10-99d9-86a435600da9)

## Data Preparation
Untuk data preparation menggunakan beberapa cara. Ada 3 dataframe yang akan diperiksa dan siapkan yaitu movie_df, rating_df, dan tags_df. Berikut penjelasan beberapa teknik yang digunakan untuk data preparationdan:

1. Removing missing value, tahapan ini diperlukan karena dengan tidak adanya missing value akan membuat performa dalam pembuatan model menjadi lebih baik. Tahapan ini dilakukan dengan code seperti berikut: dataframe.dropna(). Kode ini berfungsi untuk menghapuskan data yang memiliki null values di dalam row setiap data.

2. Normalisasi yaitu untuk mengubah nilai kolom numerik dalam kumpulan data ke skala umum, tanpa mendistorsi perbedaan dalam rentang nilai. Proses normalisasi dilakukan dengan metode Min Max. Proses tersebut dilakukan dengan code seperti gambar di bawah ini : ![Normalisasi](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/466494da-73e7-48d0-b03b-852e38dfe50b)


## Modeling
Untuk proses pemodelan disini menggunakan teknik embedding. Saya menggunakan Model [Neural Collaborative Filtering (NCF)](https://towardsdatascience.com/paper-review-neural-collaborative-filtering-explanation-implementation-ea3e031b7f96). Model Neural Collaborative Filtering (NCF) adalah jaringan saraf (neural network) yang menyediakan Collaborative Filtering berdasarkan umpan balik implisit. Secara khusus, ini memberikan rekomendasi produk berdasarkan interaksi pengguna dan item. Data pelatihan untuk model ini harus berisi urutan pasangan (ID pengguna, ID anime) yang menunjukkan bahwa pengguna yang ditentukan telah berinteraksi dengan item, misalnya, dengan memberi peringkat atau mengklik. NCF pertama kali dijelaskan oleh Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu dan Tat-Seng Chua dalam makalah [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031).

Menampilkan item movie yang user ini disukai dan tidak sebelumnya
![item movie](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/b1f8ae00-440b-42f6-a949-74a2b14cd4fd)

Berikut merupakan langkah untuk mendapatkan list rekomendasi movie berdasarkan aktivitas user berdasarkan rate yang diberikan oleh user.

1. Mencari data movie apa saja yang telah ditonton oleh user lalu memasukkannya ke dalam dataframe yang baru
2. Lalu mencari rating terendah dari movie
3. Selanjutnya membuat top_movie_refference dengan mengurutkannya berdasarkan rating dari movie.
4. Setelah itu saya membuat dataframe baru (user_pref_df) berdasarkan dataframe utama (movie_df) dan melakukan seleksi yang mana data yang dimasukkan adalah movie yang termasuk kedalam top_movie_refference
5. Dan selanjutnya menghitung rata-rata rating yang diberikan oleh user

Gambar di bawah meupakan proses penerapan dari tahapan yang saya jelaskan di atas :
![Rekomendasi](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/6f3645d6-d430-4993-b497-7314006aac4b)


Gambar di bawah ini merupakan daftar 10 rekomendasi yang dihasilkan :
![10 rekomendasi](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/cc7e2080-f5df-42cd-b18c-af4a3092fad4)

## Evaluation
Untuk bagian Evaluasi, Saya menguji performa model ini dengan mean squared error (MSE), precision, dan recall. Menurut sumber yang saya temukan, kedua metrik ini sangat cocok untuk mengukur performa model machine learning. Berikut adalah penjelasan dari setiap metrik :

* [Mean Squared Error](https://www.khoiri.com/2020/12/pengertian-dan-cara-menghitung-mean-squared-error-mse.html): Mean Squared Error (MSE) mungkin adalah fungsi loss yang paling sederhana dan paling umum, sering diajarkan dalam kursus pengantar Machine Learning. Metode Mean Squared Error secara umum digunakan untuk mengecek estimasi berapa nilai kesalahan pada peramalan. Nilai Mean Squared Error yang rendah atau nilai mean squared error mendekati nol menunjukkan bahwa hasil peramalan sesuai dengan data aktual dan bisa dijadikan untuk perhitungan peramalan di periode mendatang. Metode Mean Squared Error biasanya digunakan untuk mengevaluasi metode pengukuran dengan model regressi atau model peramalan seperti Moving Average, Weighted Moving Average dan Analisis Trendline. Cara menghitung Mean Squared Error (MSE) adalah melakukan pengurangan nilai data aktual dengan data peramalan dan hasilnya dikuadratkan (squared) kemudian dijumlahkan secara keseluruhan dan membaginya dengan banyaknya data yang ada. Nilai MSE yang didapatkan dari proyek ini adalah 0.0083
![mse 1](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/68590b49-2558-4f71-b8bc-476325d8e34c)

Di bawah ini adalah grafik mse yang dihasilkan dari proses training model yang saya buat.
![grafik mse](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/da06a763-37c6-4900-a41d-d95fb6ed0303)

* [Precision](https://dataq.wordpress.com/2013/06/16/perbedaan-precision-recall-accuracy/) : Precision adalah tingkat ketepatan antara informasi yang diminta oleh pengguna dengan jawaban yang diberikan oleh sistem. Sedangkan recall adalah tingkat keberhasilan sistem dalam menemukan kembali sebuah informasi. Nilai Precision yang didapatkan dari proyek ini adalah 1.0000 ![image](https://www.mydatamodels.com/wp-content/uploads/2020/10/5.-Precision-formula.png)
Di bawah ini adalah grafik precision yang dihasilkan dari proses training model yang saya buat.
![precision](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/ea42eb1f-78a9-4311-8fd9-64344af77f42)

* [Recall](https://dataq.wordpress.com/2013/06/16/perbedaan-precision-recall-accuracy/) : Recall adalah tingkat keberhasilan sistem dalam menemukan kembali sebuah informasi. Nilai Recall yang didapatkan dari proyek ini adalah 0.6907
Di bawah ini adalah grafik recall yang dihasilkan dari proses training model yang saya buat.
![recal](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/124fe67d-e1df-4e61-9a6a-ab62634fb45c)

