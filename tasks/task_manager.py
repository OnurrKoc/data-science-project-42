"""
DS-42 — NER ile Bilgi Çıkarımı (spaCy)
Bir haber ajansında data scientist'sin. Gelen İngilizce haber cümlelerinden
KİŞİ (PERSON), KURULUŞ (ORG), YER (GPE), PARA (MONEY) ve TARİH (DATE) gibi
varlıkları otomatik olarak çıkarman gerekiyor. Bunun için Named Entity
Recognition (NER) kullanacaksın — spaCy'nin önceden eğitilmiş
`en_core_web_sm` modeliyle.

NER, ham metni "yapılandırılmış bilgiye" çevirir: hangi cümlede hangi
kişi, kurum, yer geçiyor? Bu, bilgi çıkarımının (information extraction)
temel taşıdır.

Her fonksiyonun pass/NotImplementedError kısmını doldur. Testleri çalıştır,
hepsi geçene kadar iterate et:
`python watch.py` veya `pytest tests/test_question.py -v`

NOT: spaCy modeli (`en_core_web_sm`) zaten kurulu. Harici indirme YOK.
"""


# 1. Veriyi yükle — koda gömülü metin listesi
def load_texts():
    """
    Analiz edilecek İngilizce haber cümlelerini bir Python listesi olarak döndür.

    Harici indirme YOK — veriyi doğrudan kodda tut. En az 15 İngilizce cümle
    olmalı ve içinde bilinen PERSON / ORG / GPE / MONEY / DATE varlıkları
    geçmeli (ör. "Apple", "Tim Cook", "London", "$1 billion", "Monday").

    Returns:
        list[str]: haber cümleleri

    İpucu:
    - Fonksiyonun dışında bir NEWS_TEXTS = [...] listesi tanımla ve onu döndür.
    - Örnek cümleler:
        "Apple is looking at buying a U.K. startup for $1 billion."
        "Tim Cook announced the new iPhone in California on Monday."
        "Google opened a new office in London last year."
    """
    # TODO: koda gömülü haber cümlelerini döndür
    raise NotImplementedError


# 2. spaCy modelini yükle
def build_nlp():
    """
    spaCy'nin önceden eğitilmiş İngilizce modelini ('en_core_web_sm') yükle
    ve döndür.

    Returns:
        spacy.language.Language: yüklenmiş nlp objesi

    İpucu:
    - import spacy
    - return spacy.load("en_core_web_sm")
    """
    # TODO: spacy.load ile modeli yükle
    raise NotImplementedError


# 3. Tek metinden varlık çıkar
def extract_entities(nlp, text):
    """
    Bir metni nlp'den geçir ve içindeki varlıkları (entity) çıkar.

    Args:
        nlp: build_nlp'den dönen spaCy modeli
        text: str (tek bir cümle/metin)

    Returns:
        list[tuple]: [(varlık_metni, label), ...]
                     ör. [("Apple", "ORG"), ("London", "GPE")]

    İpucu:
    - doc = nlp(text)
    - doc.ents → metindeki varlıklar
    - Her varlık için ent.text (varlık metni) ve ent.label_ (tür) kullan
    - [(ent.text, ent.label_) for ent in doc.ents]
    """
    # TODO: nlp(text).ents üzerinden (text, label) tuple listesi üret
    raise NotImplementedError


# 4. Label bazında sayım
def count_by_label(nlp, texts):
    """
    Tüm metinlerdeki varlıkları label'a göre say.

    Args:
        nlp: spaCy modeli
        texts: list[str] (metinler)

    Returns:
        dict: {label: sayı}  ör. {"ORG": 15, "PERSON": 9, "GPE": 15, ...}

    İpucu:
    - from collections import Counter
    - Her metin için extract_entities çağır, her (text, label) için
      counter[label] += 1
    - dict(counter) döndür
    """
    # TODO: tüm metinlerde label sayımı yap, dict döndür
    raise NotImplementedError


# 5. Belirli bir label için en sık geçen n varlık
def top_entities(nlp, texts, label, n=5):
    """
    Verilen label'a (ör. "ORG") sahip varlıkları say ve en sık geçen n
    tanesini (varlık_metni, sayı) çiftleri olarak döndür.

    Args:
        nlp: spaCy modeli
        texts: list[str]
        label: str (ör. "PERSON", "ORG", "GPE")
        n: kaç varlık döndürülecek (default 5)

    Returns:
        list[tuple]: [(varlık_metni, sayı), ...] — sayıya göre azalan sıralı
                     ör. [("Google", 3), ("Apple", 2)]

    İpucu:
    - from collections import Counter
    - Her metin için extract_entities çağır, sadece label'ı eşleşen
      varlıkları say: counter[ent_text] += 1
    - return counter.most_common(n)
    """
    # TODO: label'a göre varlıkları say, most_common(n) döndür
    raise NotImplementedError


# 6. Varlık listesini label'a göre filtrele
def filter_by_label(entities, label):
    """
    Hazır bir varlık listesinden sadece istenen label'a sahip olanları süz.

    Args:
        entities: list[tuple] — extract_entities çıktısı [(text, label), ...]
        label: str (ör. "PERSON")

    Returns:
        list[tuple]: sadece label'ı eşleşen (text, label) çiftleri

    İpucu:
    - [(t, l) for (t, l) in entities if l == label]
    """
    # TODO: entities listesini label'a göre filtrele
    raise NotImplementedError


# 7. En sık geçen ORG (kuruluş)
def most_common_org(nlp, texts):
    """
    Tüm metinlerde en sık geçen KURULUŞ (ORG) varlığının ADINI döndür.

    Args:
        nlp: spaCy modeli
        texts: list[str]

    Returns:
        str: en sık geçen ORG varlığının metni (ör. "Google");
             hiç ORG yoksa None

    İpucu:
    - top_entities(nlp, texts, "ORG", n=1) → [("Google", 3)]
    - Liste boşsa None, değilse ilk tuple'ın ilk elemanını döndür.
    """
    # TODO: top_entities ile en sık ORG'u bul, sadece adını döndür
    raise NotImplementedError


# 8. Uçtan uca akış
def run_pipeline():
    """
    Uçtan uca akış:
    1. load_texts → metinleri al
    2. build_nlp → modeli yükle
    3. count_by_label → label sayımları
    4. top_entities ile en sık 3 PERSON
    5. most_common_org → en sık ORG

    Returns:
        dict: {
            'total_texts': int,
            'label_counts': dict,
            'top_persons': list[tuple] (en sık 3 PERSON),
            'most_common_org': str
        }
    """
    # TODO: yukarıdaki fonksiyonları sırayla çağırıp özet dict döndür
    raise NotImplementedError


if __name__ == "__main__":
    result = run_pipeline()
    print("📊 NER Bilgi Çıkarımı Sonuçları:")
    print(f"  Toplam metin     : {result['total_texts']}")
    print(f"  Label sayımları  : {result['label_counts']}")
    print(f"  En sık 3 PERSON  : {result['top_persons']}")
    print(f"  En sık ORG       : {result['most_common_org']}")
