"""会话主题分类。明确意图本地判定，模糊追问继承4B确认的活动主题。"""

from __future__ import annotations

import re

TOPICS = frozenset({"trip", "quote", "company", "support", "identity", "smalltalk", "general"})

_DISPLAY_NAME_PATTERNS = (
    re.compile(r"^(?:我叫|我是|叫我|请叫我|称呼我)[\s：:]*(?P<name>[^,\s，。！？!?]{1,24})$", re.I),
    re.compile(r"^(?:my name is|call me)\s+(?P<name>[A-Za-z][A-Za-z .'-]{0,30})$", re.I),
)
_INVALID_DISPLAY_NAMES = {
    "谁",
    "什么",
    "哪个",
    "一个人",
    "旅客",
    "游客",
    "中国人",
    "外国人",
    "来旅游的",
    "疯了吗",
}

_PATTERNS = (
    (
        "identity",
        re.compile(
            r"你是谁|我是谁|我叫什么|怎么称呼我|叫什么|千问|通义|qwen|"
            r"什么模型|谁(?:开发|研发|运营)",
            re.I,
        ),
    ),
    (
        "company",
        re.compile(
            r"关于我们|公司介绍|你们公司|哪家公司|所属公司|旅行社资质|"
            r"(?:旅游公司|旅行公司|旅行社).{0,12}(?:推荐|最好|靠谱|选择|哪家)|"
            r"(?:推荐|最好|靠谱|选择|哪家).{0,12}(?:旅游公司|旅行公司|旅行社)",
            re.I,
        ),
    ),
    (
        "support",
        re.compile(
            r"帮助中心|预订|订单|支付|付款|退款|取消|投诉|举报|验证码|账户|密码|"
            r"导游.{0,8}(?:骂|辱骂|威胁|骚扰|冲突|态度)|旅游纠纷|人身安全",
            re.I,
        ),
    ),
    (
        "quote",
        re.compile(r"多少钱|费用|价格|报价|预算|收费|cost|price|quote|budget|how much", re.I),
    ),
    (
        "trip",
        re.compile(
            r"旅游|旅行|攻略|景点|行程|路线|酒店|住宿|美食|门票|包车|导游|接送|"
            r"游玩|(?:\d+|[一二三四五六七八九十]+)[日天]游|"
            r"travel|trip|tour|itinerary|hotel|attraction|sightseeing",
            re.I,
        ),
    ),
    (
        "smalltalk",
        re.compile(
            r"^(?:你好|您好|嗨|谢谢|感谢|再见|你吃了吗|在吗|"
            r"你(?:再|在)?哪里|你在哪|我想打死你)[!！?？。\s]*$",
            re.I,
        ),
    ),
)

_COMPANY_RECOMMENDATION_PATTERN = re.compile(
    r"(?:旅游公司|旅行公司|旅行社|旅游平台).{0,16}(?:推荐|最好|靠谱|选择|哪家|比较)|"
    r"(?:推荐|最好|靠谱|选择|哪家|比较).{0,16}(?:旅游公司|旅行公司|旅行社|旅游平台)|"
    r"(?:旅游|旅行).{0,20}(?:推荐|最好|靠谱|选择|哪家|比较).{0,12}(?:公司|旅行社|平台)",
    re.I,
)
_NAMED_PERSON_QUERY_PATTERN = re.compile(
    r"^[\s“”\"']*[\u4e00-\u9fff·]{2,12}[\s“”\"']*(?:是谁|是什么人|什么身份)[？?。\s]*$"
)
_VERIFIED_BUSINESS_FACT_PATTERN = re.compile(
    r"法定代表人|法人|创始人|股东|董事|负责人|注册资本|成立(?:时间|日期|多久|哪年)|"
    r"旅行社资质|许可证|营业执照|注册地址",
    re.I,
)
_CONFIRMATION_FOLLOWUP_PATTERN = re.compile(
    r"^(?:需要|要|可以|好的?|行|没问题|请|请帮我|帮我|继续|是的|对)[。！!，,\s]*$",
    re.I,
)
_TRAVEL_SUPPORT_PATTERN = re.compile(
    r"导游.{0,8}(?:骂|辱骂|威胁|骚扰|冲突|态度)|"
    r"(?:投诉|举报).{0,8}(?:导游|旅行社|旅游平台)|"
    r"旅游.{0,8}(?:纠纷|投诉|人身安全|被骗)",
    re.I,
)
_GLOBAL_SUMMARY_PATTERN = re.compile(
    r"^(?:全部|全程|整个|所有|完整)?(?:对话|聊天|会话|内容)?(?:总结|汇总|回顾)|"
    r"^(?:总结|汇总|回顾)(?:全部|全程|整个|所有|完整)?(?:对话|聊天|会话|内容)?$",
    re.I,
)
_USER_QUESTION_HISTORY_PATTERN = re.compile(
    r"(?:所有|全部|今天|之前|刚才|这轮|本次).{0,12}"
    r"(?:我|用户).{0,8}(?:问过|问了|发给你|提过|说过).{0,8}(?:问题|话|内容)|"
    r"(?:我|用户).{0,8}(?:问过|问了|发给你|提过).{0,8}(?:所有|全部)(?:问题|话|内容)",
    re.I,
)
_CONTINUATION = re.compile(
    r"^(?:那|那么|然后|再|继续|还有|另外|改成|换成|加上|去掉|这个|那个|他们|老人|"
    r"孩子|第[一二三四五六七八九十\d]+天|呢|可以吗|怎么样|为什么)"
)
_LAUGHTER_PATTERN = re.compile(r"^(?:哈|呵|嘿|嘻|lol|haha)[\s哈呵嘿嘻a-z！!。,.，]*$", re.I)
_CELESTIAL_BODY_PATTERN = re.compile(
    r"月球|月亮|太阳|水星|金星|火星|木星|土星|天王星|海王星|冥王星|银河|黑洞"
)
_REAL_CELESTIAL_VENUE_PATTERN = re.compile(
    r"主题(?:乐园|公园|酒店)|博物馆|科技馆|天文馆|展览|餐厅|民宿|景区|影视城"
)
_IMPOSSIBLE_TRAVEL_PATTERN = re.compile(
    r"(?:去|到|前往|游|玩|住|飞往|一日游|二日游|两日游).{0,16}"
    r"(?:月球|月亮|太阳|水星|金星|火星|木星|土星|天王星|海王星|冥王星|银河|黑洞)|"
    r"[\u4e00-\u9fff]{1,12}的(?:月球|月亮|太阳|水星|金星|火星|木星|土星|天王星|海王星|冥王星|银河|黑洞)"
)


def classify_context(text: str, active_topic: str = "") -> str:
    value = text.strip()
    if not value:
        return active_topic if active_topic in TOPICS else "general"
    if extract_profile_facts(value):
        return "identity"
    if is_playful_or_impossible_travel_query(value) or _LAUGHTER_PATTERN.fullmatch(value):
        return "smalltalk"
    if is_company_recommendation_query(value):
        return "company"
    if _NAMED_PERSON_QUERY_PATTERN.fullmatch(value):
        return "general"
    for topic, pattern in _PATTERNS:
        if pattern.search(value):
            return topic
    if active_topic in TOPICS and (_CONTINUATION.search(value) or len(value) <= 12):
        return active_topic
    return "general"


def is_playful_or_impossible_travel_query(text: str) -> bool:
    """识别把天体当作现实目的地的玩笑，避免触发ES、图片和营销流程。"""
    value = text.strip()
    if not value or _REAL_CELESTIAL_VENUE_PATTERN.search(value):
        return False
    return bool(
        _CELESTIAL_BODY_PATTERN.search(value) and _IMPOSSIBLE_TRAVEL_PATTERN.search(value)
    )


def requests_user_display_name(text: str) -> bool:
    """只有用户明确询问自己的姓名或称呼时才向回答模型注入已保存称呼。"""
    return bool(re.search(r"我是谁|我叫什么|怎么称呼我|你记得我(?:叫|是)什么", text.strip()))


def requests_place_recall(text: str) -> bool:
    """识别用户明确要求回忆本会话提到过的地点。"""
    return bool(
        re.search(
            r"(?:之前|刚才|我们)?(?:说过|提过|提到).{0,8}(?:哪些|什么)?地方|哪些地方",
            text.strip(),
        )
    )


def is_company_recommendation_query(text: str) -> bool:
    """判断用户是否正在选择或比较旅游服务商，而非查询普通目的地。"""
    return bool(_COMPANY_RECOMMENDATION_PATTERN.search(text.strip()))


def requires_verified_business_fact(text: str) -> bool:
    """企业人物、主体和资质事实必须由业务知识库提供证据。"""
    value = text.strip()
    return bool(
        _VERIFIED_BUSINESS_FACT_PATTERN.search(value)
        or _NAMED_PERSON_QUERY_PATTERN.fullmatch(value)
    )


def business_fact_evidence_found(text: str, knowledge_context: str) -> bool:
    """姓名类查询要求资料中明确出现姓名，避免把语义相近文档当作证据。"""
    value = text.strip()
    match = _NAMED_PERSON_QUERY_PATTERN.fullmatch(value)
    if match:
        name = re.split(r"是谁|是什么人|什么身份", value, maxsplit=1)[0]
        name = name.strip(" \t\r\n“”\"'")
        return bool(name and name in knowledge_context)
    return bool(knowledge_context.strip())


def is_confirmation_followup(text: str) -> bool:
    """识别对上一轮提议的简短确认，不能把它当成新的独立问题。"""
    return bool(_CONFIRMATION_FOLLOWUP_PATTERN.fullmatch(text.strip()))


def is_travel_support_query(text: str) -> bool:
    """识别导游冲突、旅游投诉和旅途安全支持场景。"""
    return bool(_TRAVEL_SUPPORT_PATTERN.search(text.strip()))


def is_global_summary_query(text: str) -> bool:
    """识别用户明确要求总结整个会话，而不是总结当前主题。"""
    return bool(_GLOBAL_SUMMARY_PATTERN.fullmatch(text.strip()))


def is_user_question_history_query(text: str) -> bool:
    """识别要求按原文回放本会话用户问题的请求。"""
    return bool(_USER_QUESTION_HISTORY_PATTERN.search(text.strip()))


def topics_compatible(current: str, stored: str) -> bool:
    if current == stored:
        return True
    return {current, stored} <= {"trip", "quote"}


def extract_profile_facts(text: str) -> dict[str, str]:
    """提取必须跨主题保留的用户明确事实。"""
    value = text.strip()
    for pattern in _DISPLAY_NAME_PATTERNS:
        match = pattern.fullmatch(value)
        if not match:
            continue
        name = match.group("name").strip(" '“”\"")
        if name.endswith(("吗", "呢", "吧")) or any(char in name for char in "?？"):
            return {}
        if name.casefold() in {item.casefold() for item in _INVALID_DISPLAY_NAMES}:
            return {}
        return {"display_name": name}
    return {}
