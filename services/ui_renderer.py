# services/ui_renderer.py
from linebot.models import FlexSendMessage

def create_risk_flex_message(analysis_result: dict):
    """
    產出 Line Flex Message JSON
    """
    reasons_text = "\n".join([f"• {r}" for r in analysis_result['reasons']])
    
    bubble = {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": "職缺風險分析報告",
                    "weight": "bold",
                    "color": "#ffffff",
                    "size": "sm"
                },
                {
                    "type": "text",
                    "text": analysis_result['title'],
                    "weight": "bold",
                    "size": "xxl",
                    "margin": "md",
                    "color": "#ffffff"
                }
            ],
            "backgroundColor": analysis_result['color'],
            "paddingAll": "20px",
            "paddingBottom": "30px"
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": analysis_result['job_name'],
                    "weight": "bold",
                    "size": "xl",
                    "wrap": True
                },
                {
                    "type": "text",
                    "text": analysis_result['company'],
                    "size": "xs",
                    "color": "#aaaaaa",
                    "wrap": True
                },
                {
                    "type": "separator",
                    "margin": "xl"
                },
                {
                    "type": "text",
                    "text": "風險評估細項",
                    "weight": "bold",
                    "size": "sm",
                    "margin": "xl",
                    "color": "#555555"
                },
                {
                    "type": "text",
                    "text": reasons_text,
                    "size": "sm",
                    "color": "#666666",
                    "margin": "md",
                    "wrap": True,
                    "lineSpacing": "5px"
                }
            ]
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": "此分析僅供參考，求職前請再次查證",
                    "size": "xxs",
                    "color": "#aaaaaa",
                    "align": "center"
                }
            ]
        }
    }
    return FlexSendMessage(alt_text="職缺風險分析報告", contents=bubble)