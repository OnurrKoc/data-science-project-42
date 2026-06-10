import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    load_texts, build_nlp, extract_entities, count_by_label,
    top_entities, filter_by_label, most_common_org, run_pipeline,
)


# ──────────────────────────────────────────────────────
# Modül-seviye cache — testler arası tekrar model yükleme yok
# ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def texts():
    return load_texts()


@pytest.fixture(scope="module")
def nlp():
    """spaCy modelini bir kez yükle (en_core_web_sm zaten kurulu)."""
    return build_nlp()


# 1. load_texts
def test_load_texts_returns_list(texts):
    assert isinstance(texts, list)
    # En az 15 cümle, hepsi string
    assert len(texts) >= 15
    assert all(isinstance(t, str) and len(t) > 0 for t in texts)


# 2. build_nlp
def test_build_nlp_loads_model(nlp):
    # spaCy Language objesi olmalı ve çağrılabilir bir pipeline döndürmeli
    doc = nlp("Apple is in California.")
    assert hasattr(doc, "ents")


# 3. extract_entities — beklenen label'lar
def test_extract_entities_structure(nlp):
    ents = extract_entities(nlp, "Apple is looking at buying a U.K. startup for $1 billion.")
    assert isinstance(ents, list)
    # Her eleman (text, label) tuple'ı
    assert all(isinstance(e, tuple) and len(e) == 2 for e in ents)
    labels = {label for _txt, label in ents}
    # Bu cümlede ORG, GPE, MONEY beklenir
    assert "ORG" in labels
    assert "GPE" in labels
    assert "MONEY" in labels


# 4. extract_entities — doğru varlık metinleri
def test_extract_entities_values(nlp):
    ents = extract_entities(nlp, "Tim Cook announced the new iPhone in California on Monday.")
    ent_texts = [t for t, _l in ents]
    assert "Tim Cook" in ent_texts        # PERSON
    assert "California" in ent_texts       # GPE
    # PERSON label'ı Tim Cook'a bağlanmalı
    assert ("Tim Cook", "PERSON") in ents


# 5. count_by_label — PERSON/ORG/GPE sayıları > 0
def test_count_by_label(nlp, texts):
    counts = count_by_label(nlp, texts)
    assert isinstance(counts, dict)
    assert counts.get("PERSON", 0) > 0
    assert counts.get("ORG", 0) > 0
    assert counts.get("GPE", 0) > 0
    # Toplam varlık sayısı pozitif
    assert sum(counts.values()) > 0


# 6. top_entities — en sık varlığı doğru sıralar
def test_top_entities_ordering(nlp, texts):
    top_org = top_entities(nlp, texts, "ORG", n=5)
    assert isinstance(top_org, list)
    assert all(isinstance(x, tuple) and len(x) == 2 for x in top_org)
    # most_common gibi azalan sıralı olmalı
    counts = [c for _t, c in top_org]
    assert counts == sorted(counts, reverse=True)
    # Google bu metinlerde en sık geçen ORG
    assert top_org[0][0] == "Google"


# 7. top_entities — n parametresine uyar
def test_top_entities_n_limit(nlp, texts):
    top2 = top_entities(nlp, texts, "PERSON", n=2)
    assert len(top2) <= 2


# 8. filter_by_label — sadece istenen label
def test_filter_by_label(nlp):
    ents = extract_entities(nlp, "Microsoft acquired GitHub for $7.5 billion in 2018.")
    orgs = filter_by_label(ents, "ORG")
    assert isinstance(orgs, list)
    # Hepsi ORG olmalı
    assert all(label == "ORG" for _t, label in orgs)
    # Bu cümlede en az bir ORG var (Microsoft, GitHub)
    org_texts = [t for t, _l in orgs]
    assert "Microsoft" in org_texts
    # MONEY / DATE süzülmüş olmalı
    assert all(label != "MONEY" for _t, label in orgs)


# 9. most_common_org
def test_most_common_org(nlp, texts):
    org = most_common_org(nlp, texts)
    assert isinstance(org, str)
    # Google bu derlemede en sık geçen kuruluş
    assert org == "Google"


# 10. run_pipeline
def test_run_pipeline_full():
    result = run_pipeline()
    assert set(result.keys()) >= {
        'total_texts', 'label_counts', 'top_persons', 'most_common_org'
    }
    assert result['total_texts'] >= 15
    assert result['label_counts'].get("ORG", 0) > 0
    assert result['most_common_org'] == "Google"
    # En sık PERSON listesi dolu ve azalan sıralı
    persons = result['top_persons']
    assert isinstance(persons, list)
    assert len(persons) > 0
    p_counts = [c for _t, c in persons]
    assert p_counts == sorted(p_counts, reverse=True)


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────

import requests


def _send_score(user_score):
    """Kaizu API'sine skor gönder. user_id ve project_id kaizu_config'ten gelir."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return

    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return

    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "user_score": user_score,
        "is_auto": True,
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    """Tüm testleri çalıştır + skoru Kaizu'ya gönder."""
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
