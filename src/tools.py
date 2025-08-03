import os
import asyncio
import cloudscraper
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any

# 从环境变量加载用户代理，或使用默认值
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

def bing_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    使用必应搜索指定的关键词。

    Args:
        query (str): 搜索关键词。
        num_results (int): 希望返回的结果数量。

    Returns:
        一个包含搜索结果的字典列表，每个字典包含 'title', 'link', 'snippet'。
    """
    # 使用国际版 Bing (www.bing.com) 并优先请求英文结果
    search_url = f"https://www.bing.com/search?q={requests.utils.quote(query)}&setlang=en"
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'Cookie': 'SRCHHPGUSR=SRCHLANG=zh-Hans; _EDGE_S=ui=zh-cn; _EDGE_V=1'
    }
    
    try:
        # 必应搜索使用标准的 requests 库
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()  # 如果请求失败则抛出异常
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        # 尝试多个选择器以提高兼容性
        result_containers = soup.select('#b_results > li.b_algo, #b_results > .b_ans')

        for item in result_containers:
            if len(results) >= num_results:
                break

            title_element = item.select_one('h2 a')
            snippet_element = item.select_one('.b_caption p, .b_snippet')

            if title_element:
                title = title_element.get_text(strip=True)
                link = title_element.get('href')
                snippet = snippet_element.get_text(strip=True) if snippet_element else ""
                
                if title and link:
                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet
                    })
        
        # 如果主要选择器未找到结果，尝试备用方案
        if not results:
            all_links = soup.select('a')
            for link_tag in all_links:
                 if len(results) >= num_results:
                    break
                 href = link_tag.get('href')
                 text = link_tag.get_text(strip=True)
                 if href and text and href.startswith('http') and 'bing.com' not in href:
                     results.append({
                         "title": text,
                         "link": href,
                         "snippet": f"链接指向: {href}"
                     })

        return results

    except requests.RequestException as e:
        print(f"必应搜索请求出错: {e}")
        return [{"error": str(e)}]

# 创建一个 cloudscraper 实例，仅用于抓取受Cloudflare保护的网页
scraper = cloudscraper.create_scraper()

async def fetch_webpage_content(url: str) -> str:
    """
    获取并解析网页，提取主要文本内容。能处理Cloudflare保护。

    Args:
        url (str): 要抓取的网页URL。

    Returns:
        提取出的网页主要文本内容。
    """
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Referer': 'https://cn.bing.com/'
    }

    try:
        # 抓取目标网页时使用 cloudscraper
        response = await asyncio.to_thread(scraper.get, url, headers=headers, timeout=20)
        response.raise_for_status()
        
        # 使用BeautifulSoup解析
        # 使用 response.content 和 response.encoding 来确保正确解码
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding=response.encoding)
        
        # 移除脚本、样式、导航等非主要内容元素
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', '.sidebar', '.ad']):
            element.decompose()
            
        # 尝试从主要内容标签中提取文本
        main_content_selectors = ['article', 'main', '.post-content', '.entry-content', '.article-body', '#content']
        content_text = ""
        for selector in main_content_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                content_text = main_element.get_text(separator='\n', strip=True)
                break
        
        # 如果没有找到主要内容，则获取body的全部文本
        if not content_text or len(content_text) < 100:
            content_text = soup.body.get_text(separator='\n', strip=True) if soup.body else ""
            
        # 清理文本，合并多余的空行
        lines = (line.strip() for line in content_text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)

        # 添加标题
        title = soup.title.string if soup.title else "No Title Found"
        final_content = f"标题: {title}\n\n{cleaned_text}"

        # 截断过长的内容
        max_length = 8000
        if len(final_content) > max_length:
            final_content = final_content[:max_length] + "... (内容已截断)"
            
        return final_content

    except requests.exceptions.HTTPError as e:
        print(f"获取网页内容时发生HTTP状态错误: {e}")
        return f"获取网页内容失败: 服务器返回状态码 {e.response.status_code} for {e.request.url}"
    except requests.exceptions.RequestException as e:
        print(f"获取网页内容时发生请求错误: {e}")
        return f"获取网页内容失败: {e}"


def main_test():
    # 用于直接测试此模块功能的代码
    print("--- 测试搜索工具 ---")
    test_query = "Python aiohttp"
    search_results = bing_search(test_query,20)
    print(f"为 '{test_query}' 找到 {len(search_results)} 个结果:")
    for res in search_results:
        # 增加对错误情况的判断，避免KeyError
        if 'error' in res:
            print(f"搜索时发生错误: {res['error']}")
        else:
            print(f"- {res.get('title', 'N/A')}: {res.get('link', 'N/A')}")

    # 确保搜索结果有效再进行抓取
    if search_results and 'link' in search_results[0] and search_results[0]['link']:
        print("\n--- 测试网页内容抓取 ---")
        test_url = search_results[0]['link']
        print(f"正在抓取第一个结果的URL: {test_url}")
        content = fetch_webpage_content(test_url)
        print("\n抓取到的内容 (前500字符):")
        print(content[:500])

if __name__ == '__main__':
    main_test()