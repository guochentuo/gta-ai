# ruff: noqa: RUF001, RUF002
"""多语言回复规则与人工服务事件。

这些内容属于界面本地化和回复策略，不属于模型知识，也不应堆在FastAPI入口中。
"""

from __future__ import annotations

import json
import re

CONTACT_EXPLICIT_PATTERN = re.compile(
    r"怎么联系|如何联系|联系方式|联系顾问|顾问.{0,8}联系|"
    r"客服电话|客服微信|加微信|WhatsApp|LINE"
)
EXPLICIT_CONTACT_PROMPT = (
    "用户当前明确询问绿色旅行网或旅行顾问的联系方式。必须直接、简短地回应，并在自然位置"
    "原样输出一次[[GTA_CONTACT]]。不得声称无法提供联系方式，不得虚构网站、APP、电话或"
    "客服入口，不得推荐携程、飞猪、马蜂窝或其他第三方平台。联系方式由Router插入。"
)
LANGUAGE_MATCH_PROMPT = (
    "回复语言必须跟随当前这一轮用户消息，而不是跟随系统提示、历史消息或知识库资料。"
    "必须直接按用户所在地的母语习惯思考和写作，不得先生成中文再翻译。使用当地自然的"
    "词汇、语序、礼貌程度、日期时间、货币和旅游行业叫法；例如不同地区对酒店、饭店、"
    "计程车、的士、下车、落车等称呼应符合当地习惯。用户混合多种语言时，以当前问题正文"
    "的主要语言为准。品牌名、专有名词和WhatsApp、LINE、微信账号可以保持原文。"
)


_LANGUAGE_OUTPUT_RULES = {
    "zh_cn": "本轮整篇回答只能使用简体中文，并采用用户所在地区自然的中文表达。",
    "zh_tw": "本輪整篇回答只能使用繁體中文，並採用使用者所在地自然的繁體中文表達。",
    "en": "IMPORTANT: Write the entire response in English only. Do not answer in Chinese.",
    "ja": "この回答は必ず自然な日本語だけで書き、中国語では回答しないでください。",
    "ko": "이번 답변은 반드시 자연스러운 한국어로만 작성하고 중국어로 답하지 마세요.",
    "de": "Diese Antwort muss vollständig und in natürlichem Deutsch verfasst sein.",
}

_LANGUAGE_NAMES = {
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "ru": "Russian",
    "uk": "Ukrainian",
    "pl": "Polish",
    "tr": "Turkish",
    "ar": "Arabic",
    "he": "Hebrew",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
}

# 只有真正出现简繁差异字时才覆盖调用方地区提示。没有差异字的“杭州行程”仍可
# 依据zh-HK/zh-TW选择繁体, 明确写了“绿、网、们、务”等简体字时必须保持简体。
_SIMPLIFIED_CHINESE_PATTERN = re.compile(
    r"[简体台湾联系顾问务选择这里还与为会个们游费价车饭馆门开后发么来时说"
    r"对请应该点线网页资预订划画观间将让从无绿绍属谁]"
)
_TRADITIONAL_CHINESE_PATTERN = re.compile(
    r"[體臺灣聯繫顧問務選擇這裡還與為爲會個們遊費價車飯館門開後發麼麽來時"
    r"說對請應該點線網頁資預訂劃畫觀間將讓從無綠紹屬誰]"
)


def _language_output_rule(locale: str) -> str:
    if locale in _LANGUAGE_OUTPUT_RULES:
        return _LANGUAGE_OUTPUT_RULES[locale]
    language = _LANGUAGE_NAMES.get(locale, "English")
    return f"IMPORTANT: Write the entire response in natural {language} only."


def language_style_prompt(
    user_text: str = "", locale_hint: str = "", country_hint: str = ""
) -> str:
    locale = locale_hint.strip() if re.fullmatch(r"[A-Za-z0-9_-]{2,20}", locale_hint) else ""
    country = country_hint.strip() if re.fullmatch(r"[A-Za-z]{2,3}", country_hint) else ""
    output_rule = _language_output_rule(_interface_locale(user_text, locale_hint))
    if not locale and not country:
        return LANGUAGE_MATCH_PROMPT + output_rule
    return (
        LANGUAGE_MATCH_PROMPT
        + output_rule
        + f"调用方提供的用户地区信息为locale={locale or 'unknown'},"
        f"country={country.upper() or 'unknown'}；优先采用该地区的母语表达习惯。"
        "如果当前用户明确使用了不同语言或要求另一种语言，则以当前用户原话为最高优先级。"
    )


def query_explicitly_requests_contact(query: str) -> bool:
    return bool(CONTACT_EXPLICIT_PATTERN.search(query.strip()))


def _interface_locale(text: str, locale_hint: str = "") -> str:
    normalized_hint = locale_hint.replace("_", "-").lower()
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[\u0600-\u06ff]", text):
        return "ar"
    if re.search(r"[\u0590-\u05ff]", text):
        return "he"
    if re.search(r"[\u0900-\u097f]", text):
        return "hi"
    if re.search(r"[\u0e00-\u0e7f]", text):
        return "th"
    if re.search(r"[\u0400-\u04ff]", text):
        return "uk" if re.search(r"[іїєґІЇЄҐ]", text) else "ru"
    traditional_count = len(_TRADITIONAL_CHINESE_PATTERN.findall(text))
    simplified_count = len(_SIMPLIFIED_CHINESE_PATTERN.findall(text))
    if traditional_count or simplified_count:
        return "zh_tw" if traditional_count > simplified_count else "zh_cn"
    if re.search(r"[\u4e00-\u9fff]", text):
        if normalized_hint.startswith(("zh-tw", "zh-hk", "zh-mo", "zh-hant")):
            return "zh_tw"
        return "zh_cn"
    if normalized_hint.startswith("de"):
        return "de"
    if re.search(
        r"\b(?:wie|was|wann|wo|warum|reise|reisen|urlaub|kostet|tage|hotel|"
        r"sehensw[uü]rdigkeit(?:en)?|empfehl(?:en|ung))\b",
        text,
        re.IGNORECASE,
    ):
        return "de"
    latin_language_patterns = (
        ("fr", r"\b(?:voyage|voyager|combien|bonjour|séjour|visiter|itinéraire)\b"),
        ("es", r"\b(?:viaje|viajar|cuánto|hola|visitar|itinerario|turismo)\b"),
        ("pt", r"\b(?:viagem|viajar|quanto|olá|visitar|roteiro|turismo)\b"),
        ("it", r"\b(?:viaggio|viaggiare|quanto|ciao|visitare|itinerario|turismo)\b"),
        ("nl", r"\b(?:reis|reizen|hoeveel|hallo|bezoeken|route|vakantie)\b"),
        ("pl", r"\b(?:podróż|podroz|ile|cześć|zwiedzić|wycieczka|wakacje)\b"),
        ("tr", r"\b(?:seyahat|gezi|kaç|merhaba|ziyaret|tatil|turizm)\b"),
        ("vi", r"\b(?:du lịch|chuyến đi|bao nhiêu|xin chào|tham quan)\b"),
        ("id", r"\b(?:perjalanan|berapa|halo|wisata|mengunjungi|liburan)\b"),
        ("ms", r"\b(?:perjalanan|berapa|hai|pelancongan|melawat|percutian)\b"),
    )
    for language, pattern in latin_language_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return language
    primary_hint = normalized_hint.split("-", 1)[0]
    if primary_hint in _LANGUAGE_NAMES:
        return primary_hint
    if re.search(r"[A-Za-z]", text):
        return "en"
    if normalized_hint.startswith(("zh-tw", "zh-hk", "zh-mo", "zh-hant")):
        return "zh_tw"
    if normalized_hint.startswith("zh") or not text.strip():
        return "zh_cn"
    if normalized_hint.startswith("ja"):
        return "ja"
    if normalized_hint.startswith("ko"):
        return "ko"
    return "en"


_CONTACT_COPY = {
    "zh_cn": (
        "如果你愿意继续细化方案，可以在这里留下方便联系的WhatsApp、LINE或微信；"
        "也可以直接联系旅行顾问：",
        "选择你方便的方式即可。",
        "需要旅行顾问继续帮你安排吗？",
        "联系旅行顾问",
    ),
    "zh_tw": (
        "如果你願意繼續細化行程，可以在這裡留下方便聯絡的WhatsApp、LINE或微信；"
        "也可以直接聯絡旅遊顧問：",
        "選擇你方便的方式即可。",
        "需要旅遊顧問繼續幫你安排行程嗎？",
        "聯絡旅遊顧問",
    ),
    "en": (
        "If you would like to refine the itinerary, leave your preferred contact here "
        "or contact a travel consultant directly:",
        "Choose whichever channel is most convenient for you.",
        "Would you like a travel consultant to help plan the next steps?",
        "Contact a travel consultant",
    ),
    "ja": (
        "旅程をさらに詳しくご相談される場合は、ご希望の連絡先を残すか、"
        "旅行コンサルタントへ直接ご連絡ください：",
        "ご都合のよい方法をお選びください。",
        "旅行コンサルタントに旅程の続きを相談しますか？",
        "旅行コンサルタントに相談",
    ),
    "ko": (
        "여행 일정을 더 구체화하려면 편한 연락처를 남기거나 여행 상담원에게 직접 연락해 주세요:",
        "편한 방법을 선택하시면 됩니다.",
        "여행 상담원과 다음 일정을 상의하시겠어요?",
        "여행 상담원에게 문의",
    ),
    "de": (
        "Wenn du deine Reise weiter planen möchtest, hinterlasse hier deine bevorzugte "
        "Kontaktmöglichkeit oder kontaktiere direkt einen Reiseberater:",
        "Wähle einfach den für dich bequemsten Kontaktweg.",
        "Möchtest du die nächsten Schritte mit einem Reiseberater planen?",
        "Reiseberater kontaktieren",
    ),
    "fr": (
        "Pour affiner votre voyage, laissez votre moyen de contact préféré ou contactez directement un conseiller voyage :",  # noqa: E501
        "Choisissez le moyen de contact qui vous convient le mieux.",
        "Souhaitez-vous planifier la suite avec un conseiller voyage ?",
        "Contacter un conseiller",
    ),
    "es": (
        "Para seguir planificando tu viaje, deja tu contacto preferido o comunícate directamente con un asesor de viajes:",  # noqa: E501
        "Elige el medio de contacto que te resulte más cómodo.",
        "¿Quieres planificar los siguientes pasos con un asesor de viajes?",
        "Contactar a un asesor",
    ),
    "pt": (
        "Para continuar planejando sua viagem, deixe seu contato preferido ou fale diretamente com um consultor de viagens:",  # noqa: E501
        "Escolha o canal de contato mais conveniente.",
        "Gostaria de planejar os próximos passos com um consultor de viagens?",
        "Falar com um consultor",
    ),
    "it": (
        "Per continuare a pianificare il viaggio, lascia il contatto che preferisci o contatta direttamente un consulente di viaggio:",  # noqa: E501
        "Scegli il canale di contatto più comodo.",
        "Vuoi pianificare i prossimi passi con un consulente di viaggio?",
        "Contatta un consulente",
    ),
    "nl": (
        "Wil je je reis verder plannen, laat dan je favoriete contactmethode achter of neem rechtstreeks contact op met een reisadviseur:",  # noqa: E501
        "Kies het contactkanaal dat voor jou het handigst is.",
        "Wil je de volgende stappen met een reisadviseur plannen?",
        "Neem contact op",
    ),
    "ru": (
        "Чтобы продолжить планирование поездки, оставьте удобный способ связи или свяжитесь с консультантом напрямую:",  # noqa: E501
        "Выберите наиболее удобный способ связи.",
        "Хотите продолжить планирование с консультантом по путешествиям?",
        "Связаться с консультантом",
    ),
    "uk": (
        "Щоб продовжити планування подорожі, залиште зручний спосіб зв’язку або зверніться безпосередньо до консультанта:",  # noqa: E501
        "Оберіть найзручніший спосіб зв’язку.",
        "Бажаєте продовжити планування з туристичним консультантом?",
        "Зв’язатися з консультантом",
    ),
    "pl": (
        "Aby dalej zaplanować podróż, zostaw preferowany kontakt lub skontaktuj się bezpośrednio z doradcą podróży:",  # noqa: E501
        "Wybierz najwygodniejszy sposób kontaktu.",
        "Czy chcesz zaplanować kolejne kroki z doradcą podróży?",
        "Skontaktuj się z doradcą",
    ),
    "tr": (
        "Seyahatinizi ayrıntılandırmak için tercih ettiğiniz iletişim bilgisini bırakın veya doğrudan bir seyahat danışmanına ulaşın:",  # noqa: E501
        "Size en uygun iletişim kanalını seçin.",
        "Sonraki adımları bir seyahat danışmanıyla planlamak ister misiniz?",
        "Danışmana ulaşın",
    ),
    "ar": (
        "لمتابعة تخطيط رحلتك، اترك وسيلة التواصل المفضلة لديك أو تواصل مباشرة مع مستشار سفر:",
        "اختر وسيلة التواصل الأنسب لك.",
        "هل ترغب في تخطيط الخطوات التالية مع مستشار سفر؟",
        "تواصل مع مستشار",
    ),
    "he": (
        "כדי להמשיך בתכנון הטיול, השאירו את אמצעי הקשר המועדף או פנו ישירות ליועץ נסיעות:",
        "בחרו את אמצעי הקשר הנוח לכם ביותר.",
        "רוצים לתכנן את השלבים הבאים עם יועץ נסיעות?",
        "יצירת קשר עם יועץ",
    ),
    "hi": (
        "यात्रा की आगे की योजना के लिए अपना पसंदीदा संपर्क छोड़ें या सीधे यात्रा सलाहकार से संपर्क करें:",
        "अपनी सुविधा का संपर्क माध्यम चुनें।",
        "क्या आप यात्रा सलाहकार के साथ आगे की योजना बनाना चाहेंगे?",
        "यात्रा सलाहकार से संपर्क करें",
    ),
    "th": (
        "หากต้องการวางแผนการเดินทางต่อ โปรดฝากช่องทางติดต่อที่สะดวกหรือติดต่อที่ปรึกษาการเดินทางโดยตรง:",
        "เลือกช่องทางติดต่อที่สะดวกที่สุดสำหรับคุณ",
        "ต้องการวางแผนขั้นตอนต่อไปกับที่ปรึกษาการเดินทางหรือไม่?",
        "ติดต่อที่ปรึกษาการเดินทาง",
    ),
    "vi": (
        "Để tiếp tục lên kế hoạch chuyến đi, hãy để lại cách liên hệ thuận tiện hoặc liên hệ trực tiếp với tư vấn viên du lịch:",  # noqa: E501
        "Chọn kênh liên hệ thuận tiện nhất cho bạn.",
        "Bạn có muốn lên kế hoạch tiếp theo cùng tư vấn viên du lịch không?",
        "Liên hệ tư vấn viên",
    ),
    "id": (
        "Untuk melanjutkan rencana perjalanan, tinggalkan kontak pilihan Anda atau hubungi konsultan perjalanan secara langsung:",  # noqa: E501
        "Pilih saluran kontak yang paling nyaman.",
        "Ingin merencanakan langkah berikutnya bersama konsultan perjalanan?",
        "Hubungi konsultan",
    ),
    "ms": (
        "Untuk meneruskan perancangan perjalanan, tinggalkan cara hubungan pilihan anda atau hubungi perunding perjalanan secara terus:",  # noqa: E501
        "Pilih saluran hubungan yang paling mudah.",
        "Mahu merancang langkah seterusnya bersama perunding perjalanan?",
        "Hubungi perunding",
    ),
}


def contact_text_event(user_text: str = "", locale_hint: str = "") -> bytes:
    """联系方式由Router提供; 禁止模型生成或改写账号。"""
    title, description, _, _ = _CONTACT_COPY[_interface_locale(user_text, locale_hint)]
    payload = json.dumps(
        {
            "type": "contact_text",
            "title": title,
            "description": description,
            "contacts": [
                {
                    "channel": "WhatsApp",
                    "account": "+852 5555 8888",
                    "url": "https://wa.me/85255558888",
                },
                {
                    "channel": "LINE",
                    "account": "@greentourasia",
                    "url": "https://line.me/R/ti/p/@greentourasia",
                },
                {
                    "channel": "WeChat",
                    "account": "GreenTourAsia",
                    "url": "",
                },
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"data: {payload}\n\n".encode()


def handoff_offer_event(user_text: str = "", locale_hint: str = "") -> bytes:
    payload = json.loads(contact_text_event(user_text, locale_hint).decode().removeprefix("data: "))
    _, _, prompt, action_label = _CONTACT_COPY[_interface_locale(user_text, locale_hint)]
    payload["type"] = "handoff_offer"
    payload["prompt"] = prompt
    payload["action_label"] = action_label
    return (
        "data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n\n"
    ).encode()
