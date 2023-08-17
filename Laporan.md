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
[Collaborative Filtering](https://medium.com/@ranggaantok/bagaimana-sistem-rekomendasi-berkerja-e749dac64816): *collaborative filtering* adalah suatu konsep dimana opini dari pengguna lain yang ada digunakan untuk memprediksi item yang mungkin disukai/diminati oleh seorang pengguna. Algoritma *Content Based Filtering* digunakan untuk merekomendasikan film berdasarkan aktivitas pengguna pada masa lalu, sedangkan algoritma *Collaborative Filtering* digunakan untuk merekomendasikan film berdasarkan rating yang paling tinggi
Pada *collaborative filtering attribut* yang digunakan bukan konten tetapi *user behaviour*. contohnya saat merekomendasikan suatu item berdasarkan dari riwayat *rating* dari *user* tersebut maupun *user* lain.
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
| movieId | title  | genres                                    |                                                 |
|---------|--------|-------------------------------------------|-------------------------------------------------|
| 0       | 1      | Toy Story (1995)                          | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 1       | 2      | Jumanji (1995)                            | Adventure\|Children\|Fantasy                    |
| 2       | 3      | Grumpier Old Men (1995)                   | Comedy\|Romance                                 |
| 3       | 4      | Waiting to Exhale (1995)                  | Comedy\|Drama\|Romance                          |
| 4       | 5      | Father of the Bride Part II (1995)        | Comedy                                          |
| ...     | ...    | ...                                       | ...                                             |
| 9737    | 193581 | Black Butler: Book of the Atlantic (2017) | Action\|Animation\|Comedy\|Fantasy              |
| 9738    | 193583 | No Game No Life: Zero (2017)              | Animation\|Comedy\|Fantasy                      |
| 9739    | 193585 | Flint (2017)                              | Drama                                           |
| 9740    | 193587 | Bungo Stray Dogs: Dead Apple (2018)       | Action\|Animation                               |
| 9741    | 193609 | Andrew Dice Clay: Dice Rules (1991)       | Comedy                                          |


rating_df adalah isi dari dataset rating.csv yang sebelumnya telah dihilangkan kolom *timestamp* nya. 

| userId | movieId | rating |     |
|--------|---------|--------|-----|
| 0      | 1       | 1      | 4.0 |
| 1      | 1       | 3      | 4.0 |
| 2      | 1       | 6      | 4.0 |
| 3      | 1       | 47     | 5.0 |
| 4      | 1       | 50     | 5.0 |
| ...    | ...     | ...    | ... |
| 100831 | 610     | 166534 | 4.0 |
| 100832 | 610     | 168248 | 5.0 |
| 100833 | 610     | 168250 | 5.0 |
| 100834 | 610     | 168252 | 5.0 |
| 100835 | 610     | 170875 | 3.0 |

tags_df adalah isi dari dataset tags.csv.

| userId | movieId |    tag |        timestamp |            |
|-------:|--------:|-------:|-----------------:|------------|
|    0   |       2 |  60756 |            funny | 1445714994 |
|    1   |       2 |  60756 |  Highly quotable | 1445714996 |
|    2   |       2 |  60756 |     will ferrell | 1445714992 |
|    3   |       2 |  89774 |     Boxing story | 1445715207 |
|    4   |       2 |  89774 |              MMA | 1445715200 |
|   ...  |     ... |    ... |              ... |        ... |
|  3678  |     606 |   7382 |        for katie | 1171234019 |
|  3679  |     606 |   7936 |          austere | 1173392334 |
|  3680  |     610 |   3265 |           gun fu | 1493843984 |
|  3681  |     610 |   3265 | heroic bloodshed | 1493843978 |
|  3682  |     610 | 168248 | Heroic Bloodshed | 1493844270 |

Cek informasi di setiap dataset

| #   Column   Non-Null Count  Dtype  |
|-------------------------------------|
| ---  ------   --------------  ----- |
| 0   movieId  9742 non-null   int64  |
| 1   title    9742 non-null   object |
| 2   genres   9742 non-null   object |


| #   Column   Non-Null Count   Dtype   |
|---------------------------------------|
| ---  ------   --------------   -----  |
| 0   userId   100836 non-null  int64   |
| 1   movieId  100836 non-null  int64   |
| 2   rating   100836 non-null  float64 |


| #   Column     Non-Null Count  Dtype  |
|---------------------------------------|
| ---  ------     --------------  ----- |
| 0   userId     3683 non-null   int64  |
| 1   movieId    3683 non-null   int64  |
| 2   tag        3683 non-null   object |
| 3   timestamp  3683 non-null   int64  |



Cek data null Data null dapat membuat suatu hasil prediksi model menjadi tidak akurat. Cara untuk melihat apakah data ini mengandung null atau tidak adalah dengan menggunakan method dari library pandas yaitu isnull(). Berikut adalah hasil dari cek data null oleh pandas : 

| movieId | 0 |
|---------|---|
| title   | 0 |
| genres  | 0 |

| userId  | 0 |
|---------|---|
| movieId | 0 |
| rating  | 0 |


| userId    | 0 |
|-----------|---|
| movieId   | 0 |
| tag       | 0 |
| timestamp | 0 |


## Data Preparation
Untuk *data preparation* menggunakan beberapa cara. Ada 3 dataframe yang akan diperiksa dan siapkan yaitu movie_df, rating_df, dan tags_df. Berikut penjelasan beberapa teknik yang digunakan untuk data preparation:

1. Removing missing value, tahapan ini diperlukan karena dengan tidak adanya missing value akan membuat performa dalam pembuatan model menjadi lebih baik. Tahapan ini dilakukan dengan code seperti berikut: dataframe.dropna(). Kode ini berfungsi untuk menghapuskan data yang memiliki null values di dalam row setiap data.

2. Normalisasi yaitu untuk mengubah nilai kolom numerik dalam kumpulan data ke skala umum, tanpa mendistorsi perbedaan dalam rentang nilai. Proses normalisasi dilakukan dengan metode Min Max. 


## Modeling

# Pembahasan Jenis Sistem Rekomendasi dan Algoritma yang Digunakan

1. Content Based Filtering merupakan teknik dalam sistem rekomendasi yang menggunakan nilai fitur dalam data atau item sebagai dasar dalam pemberian rekomendasi. Metode ini akan mengekstrak informasi yang terdapat pada item kemudian membandingkannya dengan informasi item yang pernah dilihat atau disukai oleh *user*.

Kelebihan Content Based Filtering:

Model Content Based Filtering tidak memerlukan data pengguna lain, karena rekomendasi jenis ini diberikan berdasarkan data yang berasal oleh pengguna itu sendiri.
Model Content Based Filtering dapat digunakan untuk mengetahui preferensi atau minat spesifik pengguna, dan dapat merekomendasikan item khusus yang mungkin sangat diminati oleh beberapa pengguna lain.
Kekurangan Content Based Filtering:

Pengembangan model Content Based Filtering memerlukan pengetahuan mengenai domain atau bidang terkait.
Model Content Based Filtering hanya dapat membuat rekomendasi berdasarkan minat pengguna itu sendiri. Dengan kata lain, model memiliki kemampuan terbatas untuk memperluas minat pengguna tersebut.
Dalam proyek ini, metode yang digunakan dalam pemberian rekomendasi dengan Content Based Filtering adalah kombinasi Term Frequency – Inverse Document Frequency (TF-IDF) dengan Cosine Similarity. TF-IDF berfungsi untuk memberikan pembobotan terhadap data, dan Cosine Similiarity digunakan untuk membandingkan kemiripan suatu data dengan data lainnya berdasarkan hasil pembobotan dari TF-IDF.

1.1 Term Frequency – Inverse Document Frequency (TF-IDF)

Algoritma Term Frequency – Inverse Document Frequency (TF-IDF) merupakan algoritma yang berasal dari bidang information retrieval, yang biasanya digunakan dalam perbandingan dokumen. Algoritma ini digunakan untuk menentukan bobot dari suatu kata (t) pada suatu dokumen (d) [8].

Term Frequency (TF) merupakan bobot dari suatu kata (t) dalam dokumen (d), yang ditentukan dengan melihat jumlah kemunculan kata dalam suatu dokumen. Untuk mengurangi efek dari kata yang frekuensi kemunculannya terlalu tinggi, Inverse Document Frequency (IDF) dapat digunakan untuk mengurangi bobot dari kata dengan frekuensi kolektif (frekuensi total kemunculan kata di semua dokumen) yang tinggi. Oleh karena itu, semakin banyak frekuensi kemunculan kata, maka nilai bobot menjadi semakin rendah [8].

Kelebihan TF-IDF:

Komputasi bersifat mudah.
Dapat dengan mudah mengekstrak kata-kata yang paling deskriptif dalam suatu dokumen
Kekurangan TF-IDF:

TF-IDF menghitung kesamaan dokumen secara langsung di word-count space, yang mungkin lambat jika jumlah kata unik sangat banyak.
TF-IDF mengasumsikan bahwa jumlah kata yang berbeda memberikan pembukti kesamaan yang independen.
TF-IDF tidak menggunakan kesamaan semantik antara kata-kata.

2. Collaborative Filtering
*Collaborative Filtering* merupakan teknik dalam sistem rekomendasi yang menggunakan opini atau rating dari pengguna lain untuk memprediksi suatu item yang mungkin merupakan preferensi dari pengguna tersebut. *Collaborative Filtering* dapat dibagi menjadi dua bagian, yaitu memory-based dan model-based. Teknik memory-based dapat dibagi menjadi dua jenis, yaitu: user-based dan item-based. Teknik model-based menggunakan metode seperti Matrix factorization, Neural network, dll untuk melatih model.

Kelebihan *Collaborative Filtering*:

Pengembangan model *Collaborative Filtering* tidak memerlukan pengetahuan mengenai domain atau bidang terkait
Model *Collaborative Filtering* dapat membantu pengguna dalam menentukan minat baru sesuai dengan penilaian atau opini pengguna lain.
Kekurangan *Collaborative Filtering*:

Tidak dapat menangani item baru (cold start)
Sulit untuk menyertakan fitur sampingan untuk kueri/item selain fitur penilaian.
Kualitas rekomendasi dari sistem rekomendasi *Collaborative Filtering* bergantung pada opini pengguna lain terhadap suatu item. Opini atau rating pengguna lain dapat dianggap sebagai neighbor. Untuk meningkatkan kualitas rekomendasi sistem rekomendasi tersebut, upaya yang dapat dilakukan adalah dengan melakukan reduksi neighbor, yaitu memotong neighbor hingga ditemukan beberapa pengguna yang memiliki kesamaan (similiarity) tertinggi.

Untuk proses pemodelan disini menggunakan teknik embedding. Model machine learning yang akan dibuat adalah model sistem rekomendasi Content Based Filtering dan Collaborative Filtering. Dengan menggunakan Model [Neural Collaborative Filtering (NCF)](https://towardsdatascience.com/paper-review-neural-collaborative-filtering-explanation-implementation-ea3e031b7f96). Model Neural Collaborative Filtering (NCF) adalah jaringan saraf (neural network) yang menyediakan Collaborative Filtering berdasarkan umpan balik implisit. Secara khusus, ini memberikan rekomendasi produk berdasarkan interaksi pengguna dan item. Data pelatihan untuk model ini harus berisi urutan pasangan (ID pengguna, ID anime) yang menunjukkan bahwa pengguna yang ditentukan telah berinteraksi dengan item, misalnya, dengan memberi peringkat atau mengklik. NCF pertama kali dijelaskan oleh Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu dan Tat-Seng Chua dalam makalah [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031).

Menampilkan item movie yang user ini disukai dan tidak sebelumnya

| User  | 175    |                                                |                         |
|-------|--------|------------------------------------------------|-------------------------|
|       | rating | title                                          | genres                  |
| 91934 | 5.0    | Step Up (2006)                                 | Drama\|Romance          |
| 91932 | 5.0    | Take the Lead (2006)                           | Drama                   |
| 41344 | 5.0    | Chasing Liberty (2004)                         | Comedy\|Romance         |
| 91918 | 5.0    | Raise Your Voice (2004)                        | Romance                 |
| 91058 | 5.0    | Ice Princess (2005)                            | Children\|Comedy\|Drama |
| 39514 | 5.0    | Goal! The Dream Begins (Goal!) (2005)          | Drama                   |
| 63952 | 5.0    | Center Stage (2000)                            | Drama\|Musical          |
| 71367 | 5.0    | Footloose (1984)                               | Drama                   |
| 78029 | 4.5    | Pay It Forward (2000)                          | Drama                   |
| 88137 | 4.5    | Cinderella Story, A (2004)                     | Comedy\|Romance         |
| 78693 | 4.0    | Marie Antoinette (2006)                        | Drama\|Romance          |
| 84348 | 4.0    | Step Up 2 the Streets (2008)                   | Drama\|Musical\|Romance |
| 42139 | 4.0    | 27 Dresses (2008)                              | Comedy\|Romance         |
| 40185 | 4.0    | Producers, The (1968)                          | Comedy                  |
| 91910 | 3.5    | You Got Served (2004)                          | Drama\|Musical          |
| 91914 | 3.5    | Raising Helen (2004)                           | Comedy\|Drama\|Romance  |
| 41709 | 3.5    | Lot Like Love, A (2005)                        | Comedy\|Drama\|Romance  |
| 91919 | 3.5    | Me and You and Everyone We Know (2005)         | Comedy\|Drama           |
| 26682 | 3.5    | Secret Garden, The (1993)                      | Children\|Drama         |
| 58092 | 0.5    | Mystery Science Theater 3000: The Movie (1996) | Comedy\|Sci-Fi          |
| 91903 | 0.5    | Meteor (1979)                                  | Sci-Fi                  |
| 91906 | 0.5    | Silent Running (1972)                          | Drama\|Sci-Fi           |
| 91923 | 0.5    | Stay (2005)                                    | Thriller                |
| 91927 | 0.5    | When a Stranger Calls (2006)                   | Horror\|Thriller        |


Berikut merupakan langkah untuk mendapatkan list rekomendasi movie berdasarkan aktivitas user berdasarkan rate yang diberikan oleh user.

1. Mencari data film apa saja yang telah ditonton oleh *user* lalu memasukkannya ke dalam dataframe yang baru
2. Lalu mencari rating terendah dari film
3. Selanjutnya membuat top_movie_refference dengan mengurutkannya berdasarkan *rating* dari film.
4. Setelahnya membuat dataframe baru (user_pref_df) berdasarkan dataframe utama (movie_df) dan melakukan seleksi yang mana data yang dimasukkan adalah film yang termasuk kedalam top_movie_refference
5. Dan selanjutnya menghitung rata-rata rating yang diberikan oleh user

Gambar di bawah meupakan proses penerapan dari tahapan yang dijelaskan di atas :

|   | similar_users |                                        similarity |
|---|---------------|--------------------------------------------------:|
| 0 | 100           | [0.19105974, 0.19109485, 0.19133145, 0.1957566... |
| 1 | 382           | [0.19105974, 0.19109485, 0.19133145, 0.1957566... |
| 2 | 424           | [0.19105974, 0.19109485, 0.19133145, 0.1957566... |
| 3 | 478           | [0.19105974, 0.19109485, 0.19133145, 0.1957566... |
| 4 | 524           | [0.19105974, 0.19109485, 0.19133145, 0.1957566... |
| 5 | 302           | [0.19105974, 0.19109485, 0.19133145, 0.1957566... |
| 6 | 498           | [0.19105974, 0.19109485, 0.19133145, 0.1957566... |
| 7 | 476           | [0.19105974, 0.19109485, 0.19133145, 0.1957566... |
| 8 | 161           | [0.19105974, 0.19109485, 0.19133145, 0.1957566... |
| 9 | 479           | [0.19105974, 0.19109485, 0.19133145, 0.1957566... |


Gambar di bawah ini merupakan daftar 10 rekomendasi yang dihasilkan :

|      | movieId | title                                 | genres                  |
|------|---------|---------------------------------------|-------------------------|
| 2684 | 3594    | Center Stage (2000)                   | Drama\|Musical          |
| 2834 | 3791    | Footloose (1984)                      | Drama                   |
| 4812 | 7169    | Chasing Liberty (2004)                | Comedy\|Romance         |
| 5345 | 8911    | Raise Your Voice (2004)               | Romance                 |
| 5829 | 32289   | Ice Princess (2005)                   | Children\|Comedy\|Drama |
| 6023 | 38388   | Goal! The Dream Begins (Goal!) (2005) | Drama                   |
| 6167 | 44613   | Take the Lead (2006)                  | Drama                   |
| 6265 | 47382   | Step Up (2006)                        | Drama\|Romance          |

## Evaluation
Untuk bagian Evaluasi, Hal pertama adalah menguji performa model ini dengan mean squared error (MSE), precision, dan recall. Menurut sumber dari internet, kedua metrik ini sangat cocok untuk mengukur performa model machine learning. Berikut adalah penjelasan dari setiap metrik :

* [Mean Squared Error](https://www.khoiri.com/2020/12/pengertian-dan-cara-menghitung-mean-squared-error-mse.html): Mean Squared Error (MSE) mungkin adalah fungsi loss yang paling sederhana dan paling umum, sering diajarkan dalam kursus pengantar Machine Learning. Metode Mean Squared Error secara umum digunakan untuk mengecek estimasi berapa nilai kesalahan pada peramalan. Nilai Mean Squared Error yang rendah atau nilai mean squared error mendekati nol menunjukkan bahwa hasil peramalan sesuai dengan data aktual dan bisa dijadikan untuk perhitungan peramalan di periode mendatang. Metode Mean Squared Error biasanya digunakan untuk mengevaluasi metode pengukuran dengan model regressi atau model peramalan seperti Moving Average, Weighted Moving Average dan Analisis Trendline. Cara menghitung Mean Squared Error (MSE) adalah melakukan pengurangan nilai data aktual dengan data peramalan dan hasilnya dikuadratkan (squared) kemudian dijumlahkan secara keseluruhan dan membaginya dengan banyaknya data yang ada. Nilai MSE yang didapatkan dari proyek ini adalah 0.0083


Di bawah ini adalah grafik mse yang dihasilkan dari proses training model yang telah buat.

![grafik mse](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/da06a763-37c6-4900-a41d-d95fb6ed0303)


* [Precision](https://dataq.wordpress.com/2013/06/16/perbedaan-precision-recall-accuracy/) : Precision adalah tingkat ketepatan antara informasi yang diminta oleh pengguna dengan jawaban yang diberikan oleh sistem. Sedangkan recall adalah tingkat keberhasilan sistem dalam menemukan kembali sebuah informasi. Nilai Precision yang didapatkan dari proyek ini melalui model keras adalah 1.0000 ![image](https://www.mydatamodels.com/wp-content/uploads/2020/10/5.-Precision-formula.png)

Di bawah ini adalah grafik precision yang dihasilkan dari proses training model yang telah buat.

![precision](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/ea42eb1f-78a9-4311-8fd9-64344af77f42)


* [Recall](https://dataq.wordpress.com/2013/06/16/perbedaan-precision-recall-accuracy/) : Recall adalah tingkat keberhasilan sistem dalam menemukan kembali sebuah informasi. Nilai Recall yang didapatkan dari proyek ini adalah 0.6907

Di bawah ini adalah grafik recall yang dihasilkan dari proses training model yang telah buat.

![recal](https://github.com/aldiansah/MLT_Dicoding/assets/41302881/124fe67d-e1df-4e61-9a6a-ab62634fb45c)

