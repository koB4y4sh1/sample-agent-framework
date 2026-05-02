import asyncio

from agent_framework import Agent, Content, Message
from agent_framework.foundry import AnthropicFoundryClient, FoundryChatClient
from azure.identity import AzureCliCredential, get_bearer_token_provider


# 📝 Anthropicでの推論の引継ぎは、全リソースで共有です。 
# Foundry は Anthropic Endpoint を提供するだけのため、実際には Anthropic プラットフォーム上での推論の引継ぎが行われます。
# 詳細は https://platform.claude.com/docs/ja/build-with-claude/extended-thinking 「思考の暗号化」を参照してください。
async def run_anthropic():
    agent = Agent(
        client=AnthropicFoundryClient(
            model="claude-haiku-4-5", 
            azure_ad_token_provider=get_bearer_token_provider( 
                AzureCliCredential(),
                "https://ai.azure.com/.default",
            )
        ),
        name="AnthropicAgent",
        instructions="You are a helpful weather agent.",
    )
    messages = [
        Message(
            role="user",
            contents=[
                Content.from_text(text="東京の天気を教えて"),
            ]
        ),
        Message(
            role="assistant",
            contents=[
                Content.from_text_reasoning(
                    text="**Formulating a short introduction**\n\nI need to introduce myself in under 8 words. First, I tried \"ChatAPI: Helpful AI offering concise answers,\" which totals 6 words. Then I revised it to \"ChatAPI: Your helpful, concise AI assistant,\" which is also 6 words. Finally, I thought of \"I'm ChatAPI: concise, insightful, helpful AI,\" which is 7 words. I could also go with \"I'm ChatAPI: your helpful, insightful AI\" for a total of 7 words.",
                    protected_data="EvYCClsIDBgCKkD1evAQ1VvABWxJ2i7tlcKfg0RZJmxAerIf5VN+G5CS+r8i8Tpz/QX4h3soJUyNy+NmQ/333yIHBtQHFbmuiWZ7MhFjbGF1ZGUtc29ubmV0LTQtNjgAEgz2xtMzh6xmJBCWqCUaDPU2qYnq1AS7ouWrsCIwn9FE3reGecsH//YiTJRWXv2xejQM8Q23BnbF0uBZL5kQMrX9q08YKOc3aAmEgiY+KsgBBg5ubYLf75g9UkiooGE50VWmJi2UbkTFCSmpBge3yOw25/0GgrwM0bABqPcFr9ywll5BHE+kksZOnUUxLrMjAzzEF4r17URaVE9WEY/9uC+MTd+JZGWUzQ5PSpuWId+Ogya2W6ymz6shbR+gNG1tPvI5Itjv9WJphTvLa/dkybYdNn0EELhfanZqyc1vVpga7oFcMTgnzuAPFT5/T11Pllmw4QN8FpjHZ+9vSFZhopHDGyJQiEK3K65USHoMxUF4PR8syLhOvKsYAQ==",
                ),
                Content.from_text(text="東京は晴れで25度、大阪は曇りで22度です。"),
            ]
        ),
        Message(
            role="user",
            contents=[
                Content.from_text(text="ありがとうございます！"),
            ]
        )
    ]
    result = await agent.run(messages)
    print(f"[Anthropic]\n{result.text}")

# 📝 OpenAIでの推論の引継ぎは、同一リソース内でのみ有効です。 
# Foundry 1 -> Foundry 2 のようにリソースを跨いで推論を引き継ごうとすると、推論情報が見つからないため、APIエラーが発生します。

async def run_foundry():
    agent = Agent(
        client=FoundryChatClient(model="gpt-5.4-nano-2", credential=AzureCliCredential()),
        name="FoundryAgent",
        instructions="You are a helpful weather agent.",
    )
    messages = [
        Message(
            role="user",
            contents=[
                Content.from_text(text="どのような思考過程でその結論に至ったのですか？"),
            ]
        ),
        Message(
            role="assistant",
            contents=[
                # 📝 推論項目は、同じレスポンス内で関数呼び出しの直前に位置する場合にのみ入力として有効です。
                # テキストレスポンス（つまり、同じメッセージ内に関数呼び出しがない場合）の直前に推論項目を含めると、APIエラーが発生します。
                # 400エラー「推論項目は、必要な後続項目なしで提供されました。」

                # ❌ この推論は同じメッセージ内にfunction_callがないためMAFにより除外される
                Content.from_text_reasoning(
                    id="rs_0ba1fd54727e89250169df7e76f2ac8197a8674668c1b0f118",
                    text="**Formulating a short introduction**\n\nI need to introduce myself in under 8 words. First, I tried \"ChatAPI: Helpful AI offering concise answers,\" which totals 6 words. Then I revised it to \"ChatAPI: Your helpful, concise AI assistant,\" which is also 6 words. Finally, I thought of \"I'm ChatAPI: concise, insightful, helpful AI,\" which is 7 words. I could also go with \"I'm ChatAPI: your helpful, insightful AI\" for a total of 7 words.",
                    additional_properties={"encrypted_content":"gAAAAABp3362RC-5A-ldoXPl9yFNta_dIUdQIJmrhPtHuQa6SciYrFwOwRhJCmQ4EC2Ogx8favaPXDRRhklUEkMaNxpTjOjd5lUj2iuSTs2wrcutQixJtOxWS3PDp0jd-SfsETI0hAxU3-3QaKKSAdg5LBTr37DfnmT9ym6QY5qesuZsdv6TXN19c5CRbX8FXLbMmaBGHPvuLttxjKeY_bWplXB4cOevVJCJ8fBoNT_vu77kOUL_RYtR9vdqie3s7Je-snzOO4lp4vKLSwyYTdDrJP9eAEclzMeJtyNtx0r6u5bO3vMggDzhscj4RN1zFpKJ7oQw1Zj2fy7h7Ar4tMJluWRoDkyQPUBBVAI5m4cUGVUpomFUwzaipzxOdWukAj7SxCcgUESvR68BuTH6jE520ML85doG6YQvYehm01K34KkF8NNEcMof2NXzxpCtuqcJIpQDKHAcrZGC0R4OWTbgoXZ3rNpIBoF35tDfN7w8Dw8WSlfEDvacRC1Saz4Uc_OSHVZuxTPXcByaeqOF8ppohx0XtiLjQIgLLzphgxMlxdfu1elyAJqlY7upt3ndj8aiopbfL-b6rtzLYYQ3cdNBJVd8gPhf57aHlQ56p0a04Be8wqvUMe_Dks_pB_3Y4U-a2rGVMpqEvDcgWGkXVOEWMQNODBdcFvu2iIGFAl-FCbJ72SlH20VoAFmJ1IctEi_wjLrLPlopbM_Z9wVdya2CB6kZCednlBXPL92ZBEzbcQRlGd4VD8X4dFLpwJgDq6fDAYry0lThGmjyNK0je3nIAX7WN9WSzZQNRnzj5wx2RA4bC3q-vv7x9ExAB9FQ1Lcw_eMcUCBuYjA5giQzk0E3izTGbQZz0A=="}
                ),
                Content.from_text(text="結論：**今回は「海が好き」という条件から、海沿いで天気も参照しやすい都市としてサンディエゴ（San Diego, CA）**を選び直し、天気を調べました。"),
            ]
        ),
        Message(
            role="user",
            contents=[
                Content.from_text(text="もう一度よく考えてください。"),
            ]
        ),
        Message(
            role="assistant",
            contents=[
                
                # ✅ 後続にfunction_callがあるため、reasoningとして会話履歴に含まれる
                Content.from_text_reasoning(
                    id="rs_0f997782e6656a350169df7eb6ea3081948e4508e3761a6f07", # id 必須
                    text="**Formulating a short introduction**\n\nI need to introduce myself in under 8 words. First, I tried \"ChatAPI: Helpful AI offering concise answers,\" which totals 6 words. Then I revised it to \"ChatAPI: Your helpful, concise AI assistant,\" which is also 6 words. Finally, I thought of \"I'm ChatAPI: concise, insightful, helpful AI,\" which is 7 words. I could also go with \"I'm ChatAPI: your helpful, insightful AI\" for a total of 7 words.",
                    additional_properties={"encrypted_content":"gAAAAABp3362RC-5A-ldoXPl9yFNta_dIUdQIJmrhPtHuQa6SciYrFwOwRhJCmQ4EC2Ogx8favaPXDRRhklUEkMaNxpTjOjd5lUj2iuSTs2wrcutQixJtOxWS3PDp0jd-SfsETI0hAxU3-3QaKKSAdg5LBTr37DfnmT9ym6QY5qesuZsdv6TXN19c5CRbX8FXLbMmaBGHPvuLttxjKeY_bWplXB4cOevVJCJ8fBoNT_vu77kOUL_RYtR9vdqie3s7Je-snzOO4lp4vKLSwyYTdDrJP9eAEclzMeJtyNtx0r6u5bO3vMggDzhscj4RN1zFpKJ7oQw1Zj2fy7h7Ar4tMJluWRoDkyQPUBBVAI5m4cUGVUpomFUwzaipzxOdWukAj7SxCcgUESvR68BuTH6jE520ML85doG6YQvYehm01K34KkF8NNEcMof2NXzxpCtuqcJIpQDKHAcrZGC0R4OWTbgoXZ3rNpIBoF35tDfN7w8Dw8WSlfEDvacRC1Saz4Uc_OSHVZuxTPXcByaeqOF8ppohx0XtiLjQIgLLzphgxMlxdfu1elyAJqlY7upt3ndj8aiopbfL-b6rtzLYYQ3cdNBJVd8gPhf57aHlQ56p0a04Be8wqvUMe_Dks_pB_3Y4U-a2rGVMpqEvDcgWGkXVOEWMQNODBdcFvu2iIGFAl-FCbJ72SlH20VoAFmJ1IctEi_wjLrLPlopbM_Z9wVdya2CB6kZCednlBXPL92ZBEzbcQRlGd4VD8X4dFLpwJgDq6fDAYry0lThGmjyNK0je3nIAX7WN9WSzZQNRnzj5wx2RA4bC3q-vv7x9ExAB9FQ1Lcw_eMcUCBuYjA5giQzk0E3izTGbQZz0A=="}
                ),
                Content.from_function_call(
                    call_id="call_JpIWjXOvNhsPePYFQjyCJgfB", 
                    name="get_weather", 
                    arguments="{\"location\": \"San Diego, CA\"}"
                ),
                Content.from_function_call(
                    call_id="call_JpIWjXOvNhsPePYFQjyCJgfB", 
                    name="get_weather", 
                    arguments="{\"location\": \"San Diego, CA\"}"
                ),
            ]
        ),
        Message(
            role="tool",
            contents=[
                Content.from_function_result(call_id="call_JpIWjXOvNhsPePYFQjyCJgfB", result="{\"location\": \"San Diego, CA\", \"weather\": \"rainy\", \"temperature\": \"approx. 16C\"}"),
                Content.from_function_result(call_id="call_JpIWjXOvNhsPePYFQjyCJgfB", result="{\"location\": \"San Diego, CA\", \"weather\": \"rainy\", \"temperature\": \"approx. 18C\"}"),
            ]
        ),
        Message(
            role="assistant",
            contents=[
                Content.from_text(text="結論：**今回は「海が好き」という条件から、海沿いで天気も参照しやすい都市としてサンディエゴ（San Diego, CA）**を選び直し、天気を調べました。",),
            ]
        ),
        Message(
            role="user",
            contents=[
                Content.from_text(text="なんでサンディエゴを選んだの？"),
            ]
        )
    ]
    result = await agent.run(messages)
    print(f"[Foundry]\n{result.text}")



async def main():
    await run_anthropic()
    await run_foundry()


asyncio.run(main())