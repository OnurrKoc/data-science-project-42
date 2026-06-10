# DS-42 — NER ile Bilgi Çıkarımı (spaCy)

**Modül**: ML-04 (NLP / Bilgi Çıkarımı) • **Süre**: 2-3 saat

## 🎯 Proje Senaryosu

Bir haber ajansında **data scientist** olarak çalışıyorsun. Sisteme her gün yüzlerce İngilizce haber cümlesi düşüyor ve editörler bu metinlerden hangi **kişilerin (PERSON)**, **kuruluşların (ORG)**, **yerlerin (GPE)**, **para miktarlarının (MONEY)** ve **tarihlerin (DATE)** geçtiğini hızlıca görmek istiyor. Bunu elle yapmak imkânsız — metinleri otomatik olarak "yapılandırılmış bilgiye" çevirmen lazım.

Senin görevin: ham metinlerden varlıkları (named entities) otomatik çıkaran bir **bilgi çıkarımı (information extraction)** akışı kurmak. Doğru çalışırsa "Bu hafta en çok hangi şirket haberlere konu oldu?" ya da "Hangi kişiler en sık geçti?" gibi sorulara saniyeler içinde cevap verebilirsin.

Bunun için **NER (Named Entity Recognition / Varlık İsmi Tanıma)** kullanacaksın — spaCy'nin önceden eğitilmiş **`en_core_web_sm`** modeliyle. Model her metni bir `Doc`'a çevirir; `doc.ents` ise metinde bulunan varlıkları (her biri bir `text` + `label_` ile) verir. Sen bu varlıkları sayacak, filtreleyecek ve sıralayacaksın.

> **Önemli**: Bu projede **harici veri indirilmez**. Analiz edilecek İngilizce haber cümleleri doğrudan **koda gömülüdür** ve `load_texts()` fonksiyonundan bir Python listesi olarak gelir. Tek gereksinim: spaCy `en_core_web_sm` modelinin kurulu olması (ortamda hazır gelir).

Bu projede şunları uygulayacaksın:
- ✅ **spaCy modeli yükleme** (`spacy.load("en_core_web_sm")`)
- ✅ **NER ile varlık çıkarma** (`doc.ents` → `(ent.text, ent.label_)`)
- ✅ **Varlık türleri**: PERSON / ORG / GPE / MONEY / DATE
- ✅ **Label bazında sayım** (`collections.Counter`)
- ✅ **En sık geçen varlıkları sıralama** (`most_common`)
- ✅ **Label'a göre filtreleme**
- ✅ **Uçtan uca bilgi çıkarımı akışı** (`run_pipeline`)

## 📦 Proje Kurulumu

```bash
# Fork + clone
git clone <your-fork-url>
cd data-science-project-42

# Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# Dependencies
pip install -r requirements.txt

# spaCy İngilizce modeli (kurulu değilse)
python -m spacy download en_core_web_sm

# Auto test runner (dosya değişince çalışır)
python watch.py

# Manuel test
pytest tests/test_question.py -v
```

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

Skorunun Kaizu hesabına yazılması için **`kaizu_config.py`** dosyasını aç ve **`USER_ID`** alanını kendi user_id'nle değiştir:

```python
USER_ID = 0      # ← Kaizu profilinden alıp buraya yaz
PROJECT_ID = 722 # ← Bu projeye ait, dokunma
```

User_id'ni Kaizu profilinden bulabilirsin (Profile → Settings → User ID).

Skor göndermek için tüm testleri toplu çalıştırmalısın:

```bash
python tests/test_question.py
```

Bu komut tüm testleri çalıştırır, **passed/total oranını otomatik Kaizu'ya gönderir**. Geliştirme sırasında `pytest -v` kullanmaya devam edebilirsin (skor göndermez).

## 📚 Veri — Koda Gömülü Haber Cümleleri

### Kaynak
- Veri **harici dosya değildir**. ~18 kısa İngilizce haber cümlesi `load_texts()` fonksiyonunda Python listesi olarak tutulur.
- `data/` klasörü `.gitignore`'da olduğu için zaten repo'ya gitmez — ama bu projede dosya okuması da gerekmez.

### İçerik
Her cümle içinde bilinen varlıklar geçer:

| Label | Anlamı | Örnekler |
|-------|--------|----------|
| `PERSON` | Kişi | Tim Cook, Barack Obama, Elon Musk |
| `ORG` | Kuruluş | Apple, Google, Microsoft |
| `GPE` | Ülke / şehir / yer | London, California, Japan |
| `MONEY` | Para miktarı | $1 billion, $7.5 billion |
| `DATE` | Tarih / zaman | Monday, 2018, October 2021 |

### Örnek Cümleler (kavramsal)
```
"Apple is looking at buying a U.K. startup for $1 billion."
   → ("Apple", ORG), ("U.K.", GPE), ("$1 billion", MONEY)
"Tim Cook announced the new iPhone in California on Monday."
   → ("Tim Cook", PERSON), ("California", GPE), ("Monday", DATE)
"Google opened a new office in London last year."
   → ("Google", ORG), ("London", GPE), ("last year", DATE)
```

### Model Notu
spaCy `en_core_web_sm` küçük ve önceden eğitilmiş bir İngilizce pipeline'dır. Deterministiktir: aynı model + aynı girdi → aynı varlıklar. Bu yüzden testler sabit sonuçlar bekler (ör. en sık geçen ORG: **Google**).

## 📋 Görevler (`tasks/task_manager.py`)

`task_manager.py` dosyasındaki **8 fonksiyonu** sırayla doldur. Her task altta testler pass olana kadar düzenlenmeli.

1. **`load_texts()`** — koda gömülü İngilizce haber cümlelerini liste olarak döndür (≥15)
2. **`build_nlp()`** — `spacy.load("en_core_web_sm")`
3. **`extract_entities(nlp, text)`** — bir metinden `[(text, label), ...]` çıkar
4. **`count_by_label(nlp, texts)`** — tüm metinlerde label bazında sayım → dict
5. **`top_entities(nlp, texts, label, n=5)`** — bir label için en sık n varlık → `[(text, sayı), ...]`
6. **`filter_by_label(entities, label)`** — varlık listesinden sadece istenen label'ı süz
7. **`most_common_org(nlp, texts)`** — en sık geçen ORG'un adını döndür
8. **`run_pipeline()`** — uçtan uca akış, özet dict dön

## 🎓 Öğrenme Hedefleri

Bu projeyi bitirdiğinde:
- [x] spaCy modeli yükleyip metni işleyebileceksin (`nlp(text)`)
- [x] **NER** ile metinden varlık çıkarabileceksin (`doc.ents`)
- [x] Varlık türlerini (PERSON / ORG / GPE / MONEY / DATE) ayırt edebileceksin
- [x] Varlıkları **sayabilecek** ve en sık geçenleri **sıralayabileceksin**
- [x] Label'a göre **filtreleme** yapabileceksin
- [x] Ham metni **yapılandırılmış bilgiye** çeviren bir akış kurabileceksin

## 🧪 Testler

Test dosyası: `tests/test_question.py` (10 test)

Tümü pass olmalı:
- `load_texts` ≥15 string döndürüyor mu
- `build_nlp` çalışan bir spaCy modeli döndürüyor mu
- `extract_entities` bir cümlede beklenen label'ları (ORG/GPE/MONEY) döndürüyor mu
- `extract_entities` doğru varlık metinlerini (ör. "Tim Cook" → PERSON) veriyor mu
- `count_by_label`'da PERSON / ORG / GPE sayıları > 0 mı
- `top_entities` en sık varlığı doğru sıralıyor mu (azalan)
- `filter_by_label` sadece istenen label'ı veriyor mu
- `most_common_org` doğru kuruluşu (**Google**) döndürüyor mu
- `run_pipeline` uçtan uca özet dict üretiyor mu

## 📊 Beklenen Sonuçlar

```
Toplam metin: ~18
Label sayımları: ORG ~15, GPE ~15, DATE ~13, PERSON ~9, MONEY ~5
En sık geçen ORG: Google (3 kez)
En sık PERSON listesi: azalan sıralı (text, sayı) çiftleri
```

## 💡 İpuçları

- `nlp = spacy.load("en_core_web_sm")` — modeli bir kez yükle, tekrar tekrar yükleme.
- `doc = nlp(text)` sonra `doc.ents` → varlıklar. Her varlık: `ent.text` (metin) + `ent.label_` (tür).
- Sayım için `from collections import Counter`. `counter.most_common(n)` → en sık n eleman, azalan sıralı.
- `top_entities` ve `count_by_label` aynı `extract_entities`'i tekrar kullanır — kodu DRY tut.
- `filter_by_label` hazır bir varlık listesini (extract_entities çıktısı) alır; nlp çalıştırmaz.
- Label kodları: PERSON (kişi), ORG (kuruluş), GPE (ülke/şehir), MONEY (para), DATE (tarih). spaCy etiketleri büyük harftir.

## 🚫 Dikkat

- `tests/test_question.py` dosyasını **değiştirme**
- `load_texts()` içindeki cümleleri değiştirme — testler sabit sonuçlara (ör. ORG=Google) dayanıyor
- `_solution/` klasörü yok (DB'de saklanır, dersin haftası geçince açılır)
- `data/` klasörü repo'ya **gitmez** (.gitignore'da exclude); zaten veri koda gömülü
- Dokunabileceğin **2 dosya**: `tasks/task_manager.py` (kodu yaz) + `kaizu_config.py` (sadece USER_ID)
