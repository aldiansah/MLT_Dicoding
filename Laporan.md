# Laporan Proyek Machine Learning - Aldiansah Permana

## Domain Proyek
Industri perfilman dunia merupakan salah satu industri yang tidak terpengaruh dengan maraknya hiburan digital seperti munculnya media sosial, program televisi yang beragam dan game. Industri film yang terus melakukan produksi ini semakin menambah informasi film yang melimpah di internet. Kondisi ini justru membuat para penikmat film menjadi kebingungan ketika harus  memilih film kesukaannya. Sistem rekomendasi menyediakan  informasi berdasarkan interaksi pengguna dan item yang telah terekam sebelumnya. Penelitian ini akan membahas pembangunan sistem rekomendasi dengan metode pendekatan Collaborative Filtering.


## Business Understanding
### Problem Statements
* Bagaimana cara meningkatkan user experience saat mencari movie yang ingin ditonton?
* Bagaimana cara membuat sistem rekomendasi movie menggunakan metode collaborative filtering?

### Goals
* Meningkatkan user experience saat mencari film yang ingin ditonton.
* Dapat mengimplementasikan metode collaborative filtering untuk sistem rekomendasi movie.

### Solution statements
Dataset yang digunakan hanya berisi tentang rating atau hasil penilaian pengguna dan genre film, maka solusi yang sangat tepat untuk masalah ini adalah dengan menggunakan collaborative filtering.
[Collaborative Filtering](https://medium.com/@ranggaantok/bagaimana-sistem-rekomendasi-berkerja-e749dac64816): collaborative filtering adalah suatu konsep dimana opini dari pengguna lain yang ada digunakan untuk memprediksi item yang mungkin disukai/diminati oleh seorang pengguna.
Pada collaborative filtering attribut yang digunakan bukan konten tetapi user behaviour. contohnya kita merekomendasikan suatu item berdasarkan dari riwayat rating dari user tersebut maupun user lain.
![image](https://miro.medium.com/max/335/1*O6ON-kQ34pMCYOHSr7ZebQ.png)



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
https://user-images.githubusercontent.com/41302881/261095217-adb8a255-b595-48dc-bf19-335f451cb4d1.png

rating_df adalah isi dari dataset rating.csv yang sebelumnya telah dihilangkan kolom timestamp nya.
https://user-images.githubusercontent.com/41302881/261096508-c8b82236-fe3c-4fe2-8533-b5078a3de341.png

tags_df adalah isi dari dataset tags.csv.
https://user-images.githubusercontent.com/41302881/261096871-9ece867b-e13a-40b4-b9fe-66c3c7da1073.png

Cek informasi di setiap dataset

https://user-images.githubusercontent.com/41302881/261097651-fba5475d-f23e-4a83-857a-6c609e406e3b.png

https://user-images.githubusercontent.com/41302881/261097893-84916854-f546-4b63-87c6-a9abd13cda8f.png

https://user-images.githubusercontent.com/41302881/261098271-b294cba2-250e-40e8-addc-14a8c1746cd7.png



Cek data null Data null dapat membuat suatu hasil prediksi model menjadi tidak akurat. Cara untuk melihat apakah data ini mengandung null atau tidak adalah dengan menggunakan method dari library pandas yaitu isnull(). Berikut adalah hasil dari cek data null oleh pandas : 
https://user-images.githubusercontent.com/41302881/261098639-81343358-e1b3-40ce-9537-877e8597b120.png

https://user-images.githubusercontent.com/41302881/261098918-f9f3156d-0a90-46c4-bc03-63d39cd50c57.png

https://user-images.githubusercontent.com/41302881/261099231-2261225b-ec72-430d-8c8f-720494b7d07b.png

## Data Preparation
Untuk data preparation menggunakan beberapa cara. Ada 3 dataframe yang akan diperiksa dan siapkan yaitu movie_df, rating_df, dan tags_df. Berikut penjelasan beberapa teknik yang digunakan untuk data preparationdan:

1. Removing missing value, tahapan ini diperlukan karena dengan tidak adanya missing value akan membuat performa dalam pembuatan model menjadi lebih baik. Tahapan ini dilakukan dengan code seperti berikut: dataframe.dropna(). Kode ini berfungsi untuk menghapuskan data yang memiliki null values di dalam row setiap data.

2. Normalisasi yaitu untuk mengubah nilai kolom numerik dalam kumpulan data ke skala umum, tanpa mendistorsi perbedaan dalam rentang nilai. Proses normalisasi dilakukan dengan metode Min Max. Proses tersebut dilakukan dengan code seperti gambar di bawah ini : https://user-images.githubusercontent.com/41302881/261101011-ca25bf7b-8009-4506-91d0-f130cce709b4.png


## Modeling
Untuk proses pemodelan disini menggunakan teknik embedding. Saya menggunakan Model [Neural Collaborative Filtering (NCF)](https://towardsdatascience.com/paper-review-neural-collaborative-filtering-explanation-implementation-ea3e031b7f96). Model Neural Collaborative Filtering (NCF) adalah jaringan saraf (neural network) yang menyediakan Collaborative Filtering berdasarkan umpan balik implisit. Secara khusus, ini memberikan rekomendasi produk berdasarkan interaksi pengguna dan item. Data pelatihan untuk model ini harus berisi urutan pasangan (ID pengguna, ID anime) yang menunjukkan bahwa pengguna yang ditentukan telah berinteraksi dengan item, misalnya, dengan memberi peringkat atau mengklik. NCF pertama kali dijelaskan oleh Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu dan Tat-Seng Chua dalam makalah [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031).

Menampilkan item movie yang user ini disukai dan tidak sebelumnya
https://user-images.githubusercontent.com/41302881/261102035-d579c441-f9a6-491c-a374-e0bd316cb27b.png

Berikut merupakan langkah untuk mendapatkan list rekomendasi movie berdasarkan aktivitas user berdasarkan rate yang diberikan oleh user.

1. Mencari data movie apa saja yang telah ditonton oleh user lalu memasukkannya ke dalam dataframe yang baru
2. Lalu mencari rating terendah dari movie
3. Selanjutnya membuat top_movie_refference dengan mengurutkannya berdasarkan rating dari movie.
4. Setelah itu saya membuat dataframe baru (user_pref_df) berdasarkan dataframe utama (movie_df) dan melakukan seleksi yang mana data yang dimasukkan adalah movie yang termasuk kedalam top_movie_refference
5. Dan selanjutnya menghitung rata-rata rating yang diberikan oleh user

Gambar di bawah meupakan proses penerapan dari tahapan yang saya jelaskan di atas :
https://user-images.githubusercontent.com/41302881/261102539-8e12da9d-3a92-49be-b28b-3126d4477b89.png


Gambar di bawah ini merupakan daftar 10 rekomendasi yang dihasilkan :
https://user-images.githubusercontent.com/41302881/261102762-ef88e52e-4510-4441-a39c-edbd29c162f8.png

## Evaluation
Untuk bagian Evaluasi, Saya menguji performa model ini dengan mean squared error (MSE), precision, dan recall. Menurut sumber yang saya temukan, kedua metrik ini sangat cocok untuk mengukur performa model machine learning. Berikut adalah penjelasan dari setiap metrik :

* [Mean Squared Error](https://www.khoiri.com/2020/12/pengertian-dan-cara-menghitung-mean-squared-error-mse.html): Mean Squared Error (MSE) mungkin adalah fungsi loss yang paling sederhana dan paling umum, sering diajarkan dalam kursus pengantar Machine Learning. Metode Mean Squared Error secara umum digunakan untuk mengecek estimasi berapa nilai kesalahan pada peramalan. Nilai Mean Squared Error yang rendah atau nilai mean squared error mendekati nol menunjukkan bahwa hasil peramalan sesuai dengan data aktual dan bisa dijadikan untuk perhitungan peramalan di periode mendatang. Metode Mean Squared Error biasanya digunakan untuk mengevaluasi metode pengukuran dengan model regressi atau model peramalan seperti Moving Average, Weighted Moving Average dan Analisis Trendline. Cara menghitung Mean Squared Error (MSE) adalah melakukan pengurangan nilai data aktual dengan data peramalan dan hasilnya dikuadratkan (squared) kemudian dijumlahkan secara keseluruhan dan membaginya dengan banyaknya data yang ada. Nilai MSE yang didapatkan dari proyek ini adalah 0.0083
https://user-images.githubusercontent.com/41302881/261109494-8f22d2a2-4928-44c2-8af6-a56b19fe3338.png

Di bawah ini adalah grafik mse yang dihasilkan dari proses training model yang saya buat.
https://user-images.githubusercontent.com/41302881/261109695-162dd5ba-e44a-4364-9003-b541e68cc2b9.png

* [Precision](https://dataq.wordpress.com/2013/06/16/perbedaan-precision-recall-accuracy/) : Precision adalah tingkat ketepatan antara informasi yang diminta oleh pengguna dengan jawaban yang diberikan oleh sistem. Sedangkan recall adalah tingkat keberhasilan sistem dalam menemukan kembali sebuah informasi. Nilai Precision yang didapatkan dari proyek ini adalah 1.0000 ![image](https://www.mydatamodels.com/wp-content/uploads/2020/10/5.-Precision-formula.png)
Di bawah ini adalah grafik precision yang dihasilkan dari proses training model yang saya buat.
https://user-images.githubusercontent.com/41302881/261110138-79662639-9a44-4629-a5e9-99174130e4dd.png

* [Recall](https://dataq.wordpress.com/2013/06/16/perbedaan-precision-recall-accuracy/) : Recall adalah tingkat keberhasilan sistem dalam menemukan kembali sebuah informasi. Nilai Recall yang didapatkan dari proyek ini adalah 0.6907
Di bawah ini adalah grafik recall yang dihasilkan dari proses training model yang saya buat.
https://user-images.githubusercontent.com/41302881/261110371-e6fd870b-d215-4d30-b507-df51e4a218ed.png

