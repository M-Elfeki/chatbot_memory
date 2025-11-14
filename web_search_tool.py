#!/usr/bin/env python3
"""
Web search tool for LLM with robust error handling and response parsing.
Uses DuckDuckGo for search (free, no API key required).
"""
import json
import time
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import quote_plus


# Cache for rate limit scenarios
_RATE_LIMIT_CACHE: Dict[str, Dict[str, Any]] = {}
_RATE_LIMIT_CACHE_TTL = timedelta(minutes=5)  # Cache for 5 minutes


class WebSearchError(Exception):
    """Custom exception for web search errors."""
    pass


class RateLimitError(WebSearchError):
    """Raised when rate limit is hit."""
    pass


class TimeoutError(WebSearchError):
    """Raised when search times out."""
    pass


class NoResultsError(WebSearchError):
    """Raised when no results are found."""
    pass


def _get_cached_result(query: str) -> Optional[Dict[str, Any]]:
    """Get cached result if available and not expired."""
    if query in _RATE_LIMIT_CACHE:
        cached = _RATE_LIMIT_CACHE[query]
        if datetime.now() - cached["timestamp"] < _RATE_LIMIT_CACHE_TTL:
            return cached["result"]
        else:
            del _RATE_LIMIT_CACHE[query]
    return None


def _cache_result(query: str, result: Dict[str, Any]):
    """Cache a search result."""
    _RATE_LIMIT_CACHE[query] = {
        "result": result,
        "timestamp": datetime.now()
    }


def _parse_duckduckgo_results(html_content: str, query: str) -> List[Dict[str, str]]:
    """
    Parse DuckDuckGo HTML search results.
    Extracts title, snippet, and URL from each result.
    Uses multiple parsing strategies for robustness.
    """
    from bs4 import BeautifulSoup
    
    results = []
    soup = BeautifulSoup(html_content, 'html.parser')
    seen_urls = set()  # Avoid duplicates
    
    # Strategy 1: Look for result containers with various class names
    result_containers = (
        soup.find_all('div', class_='result') +
        soup.find_all('div', class_='web-result') +
        soup.find_all('article', class_='result') +
        soup.find_all('div', class_='links_main') +
        soup.find_all('div', class_='result__body')
    )
    
    if result_containers:
        for container in result_containers[:15]:  # Check more containers
            # Try multiple ways to find title
            title_elem = (
                container.find('h2') or
                container.find('a', class_='result__a') or
                container.find('a', class_='result-link') or
                container.find('a', href=True)
            )
            
            # Try multiple ways to find URL
            url_elem = (
                container.find('a', class_='result__a', href=True) or
                container.find('a', class_='result-link', href=True) or
                container.find('a', href=True)
            )
            
            # Try multiple ways to find snippet
            snippet_elem = (
                container.find('span', class_='result__snippet') or
                container.find('div', class_='result__snippet') or
                container.find('a', class_='result__snippet') or
                container.find('p', class_='result__snippet') or
                container.find('span') or
                container.find('p')
            )
            
            if title_elem and url_elem:
                title = title_elem.get_text(strip=True)
                url = url_elem.get('href', '')
                
                # Clean URL (remove DuckDuckGo redirect)
                if url.startswith('/l/?kh=') or 'duckduckgo.com/l/' in url:
                    # Extract actual URL from redirect
                    if 'uddg=' in url:
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                        if 'uddg' in parsed:
                            url = urllib.parse.unquote(parsed['uddg'][0])
                
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                # Filter valid results
                if (title and 
                    url.startswith('http') and 
                    'duckduckgo.com' not in url and
                    url not in seen_urls):
                    seen_urls.add(url)
                    results.append({
                        "title": title[:200],
                        "snippet": snippet[:300] if snippet else "No snippet available",
                        "url": url
                    })
    
    # Strategy 2: If no structured results, try finding all external links
    if not results:
        links = soup.find_all('a', href=True)
        for link in links[:20]:
            href = link.get('href', '')
            
            # Handle DuckDuckGo redirect URLs
            if href.startswith('/l/?kh=') or 'duckduckgo.com/l/' in href:
                if 'uddg=' in href:
                    import urllib.parse
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    if 'uddg' in parsed:
                        href = urllib.parse.unquote(parsed['uddg'][0])
            
            # Filter valid external links
            if (href.startswith('http') and 
                'duckduckgo.com' not in href and
                href not in seen_urls):
                title = link.get_text(strip=True)
                if title and len(title) > 5:  # Filter out very short titles
                    # Try to find snippet in parent or sibling elements
                    snippet = ""
                    parent = link.find_parent()
                    if parent:
                        # Look for snippet in siblings or parent
                        for elem in [parent.find_next_sibling(), parent.find_previous_sibling()]:
                            if elem:
                                snippet_text = elem.get_text(strip=True)
                                if snippet_text and len(snippet_text) > 20:
                                    snippet = snippet_text
                                    break
                        # If no sibling, check parent's text
                        if not snippet:
                            parent_text = parent.get_text(strip=True)
                            if len(parent_text) > len(title) + 20:
                                snippet = parent_text[len(title):].strip()[:300]
                    
                    seen_urls.add(href)
                    results.append({
                        "title": title[:200],
                        "snippet": snippet[:300] if snippet else "No snippet available",
                        "url": href
                    })
    
    return results[:10]  # Return top 10 results


def _search_duckduckgo_html(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo using HTML scraping.
    Returns list of results with title, snippet, and URL.
    """
    # DuckDuckGo HTML search URL
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        
        # DuckDuckGo may return 202 (Accepted) - this is OK, content may still be valid
        if response.status_code not in [200, 202]:
            response.raise_for_status()
        
        # Check for rate limiting
        if response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        
        results = _parse_duckduckgo_results(response.text, query)
        return results[:max_results]
    
    except requests.exceptions.Timeout:
        raise TimeoutError("Search request timed out")
    except requests.exceptions.RequestException as e:
        if "429" in str(e) or "rate limit" in str(e).lower():
            raise RateLimitError(f"Rate limit exceeded: {e}")
        raise WebSearchError(f"Search request failed: {e}")


def _search_duckduckgo_library(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Use duckduckgo-search library (most reliable method).
    Falls back to HTML scraping if library is not available.
    """
    import warnings
    try:
        # Suppress deprecation warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # Try new package name first
            try:
                from ddgs import DDGS
            except ImportError:
                # Fall back to old package name
                from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            # Search for web results
            search_results = ddgs.text(query, max_results=max_results)
            
            # Check if we got any results
            if search_results:
                for result in search_results:
                    title = result.get("title", "")
                    url = result.get("href", "")
                    body = result.get("body", "")
                    
                    if title and url:  # Only add if we have title and URL
                        results.append({
                            "title": title[:200],
                            "snippet": body[:300] if body else "No snippet available",
                            "url": url
                        })
        
        return results
    
    except ImportError:
        # Library not available, fall back to HTML scraping
        return []
    except Exception as e:
        # If library fails, return empty list to trigger fallback
        return []


def _search_duckduckgo_api(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Try DuckDuckGo search library first, then fall back to HTML scraping.
    """
    # Try library first (most reliable)
    results = _search_duckduckgo_library(query, max_results)
    
    # If library returned results, use them
    if results:
        return results
    
    # Otherwise fall back to HTML scraping
    return _search_duckduckgo_html(query, max_results)


def web_search(
    query: str,
    max_results: int = 10,
    retry_on_timeout: bool = True,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Perform a web search with robust error handling.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 10)
        retry_on_timeout: Whether to retry once on timeout (default: True)
        use_cache: Whether to use cached results for rate limits (default: True)
    
    Returns:
        Dictionary with:
            - success: bool
            - results: List of dicts with 'title', 'snippet', 'url'
            - error: Optional error message
            - cached: bool indicating if result was from cache
    
    Raises:
        WebSearchError: Base exception for search errors
        RateLimitError: When rate limit is hit
        TimeoutError: When search times out after retries
        NoResultsError: When no results are found
    """
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty")
    
    # Check cache first (for rate limit scenarios)
    if use_cache:
        cached = _get_cached_result(query)
        if cached:
            return {**cached, "cached": True}
    
    # Try search with retry logic
    last_error = None
    for attempt in range(2 if retry_on_timeout else 1):
        try:
            results = _search_duckduckgo_api(query, max_results)
            
            if not results:
                # No results found
                error_msg = f"No results found for query: {query}"
                result = {
                    "success": False,
                    "results": [],
                    "error": error_msg,
                    "cached": False
                }
                
                # Cache the "no results" response briefly
                if use_cache:
                    _cache_result(query, result)
                
                raise NoResultsError(error_msg)
            
            # Success
            result = {
                "success": True,
                "results": results,
                "error": None,
                "cached": False
            }
            
            # Cache successful results
            if use_cache:
                _cache_result(query, result)
            
            return result
        
        except RateLimitError as e:
            # Rate limit hit - return cached "search unavailable" if available
            if use_cache:
                cached = _get_cached_result(query)
                if cached:
                    return {**cached, "cached": True}
                
                # Cache a "search unavailable" response
                error_result = {
                    "success": False,
                    "results": [],
                    "error": "Search unavailable due to rate limiting. Please try again later.",
                    "cached": False
                }
                _cache_result(query, error_result)
                return {**error_result, "cached": True}
            
            raise
        
        except TimeoutError as e:
            last_error = e
            if attempt == 0 and retry_on_timeout:
                # Wait a bit before retry
                time.sleep(1)
                continue
            # After retry or if retry disabled, fail gracefully
            error_result = {
                "success": False,
                "results": [],
                "error": f"Search timed out: {str(e)}",
                "cached": False
            }
            return error_result
        
        except NoResultsError:
            raise
        
        except WebSearchError as e:
            # Other search errors - fail gracefully
            error_result = {
                "success": False,
                "results": [],
                "error": f"Search failed: {str(e)}",
                "cached": False
            }
            return error_result
    
    # Should not reach here, but handle timeout after retries
    if last_error:
        error_result = {
            "success": False,
            "results": [],
            "error": f"Search timed out after retry: {str(last_error)}",
            "cached": False
        }
        return error_result
    
    raise WebSearchError("Unexpected error in web search")


def web_search_executor(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Tool executor function for use with generate_with_tools.
    Converts search results to a formatted string that encourages summarization.
    
    Args:
        tool_name: Should be "web_search"
        arguments: Dictionary with 'query' key and optional 'max_results'
    
    Returns:
        Formatted string with search results or error message
    """
    if tool_name != "web_search":
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    # Validate arguments - must have 'query' parameter
    if not isinstance(arguments, dict):
        return json.dumps({"error": "Invalid arguments: must be a dictionary"})
    
    query = arguments.get("query", "")
    # Check for invalid argument patterns
    if not query or not isinstance(query, str) or len(query.strip()) == 0:
        # Check if arguments contain invalid keys like 'cursor', 'id' without 'query'
        if 'cursor' in arguments or ('id' in arguments and 'query' not in arguments):
            return json.dumps({
                "error": "Invalid arguments: 'query' parameter is required. "
                         "Do not use 'cursor' or 'id' parameters."
            })
        return json.dumps({"error": "Missing required parameter 'query'"})
    
    max_results = arguments.get("max_results", 10)
    if not isinstance(max_results, int) or max_results < 1:
        max_results = 10
    
    try:
        result = web_search(query, max_results=max_results)
        
        if result["success"]:
            # Format results in a more readable way that encourages summarization
            results = result["results"]
            formatted_results = []
            
            for i, res in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. {res['title']}\n"
                    f"   URL: {res['url']}\n"
                    f"   {res['snippet']}"
                )
            
            # Return formatted string - don't include instruction to avoid echo
            # The LLM should naturally summarize based on the results provided
            return (
                f"Found {len(results)} search results for '{query}':\n\n"
                + "\n\n".join(formatted_results)
            )
        else:
            return f"Search failed: {result['error']}"
    
    except NoResultsError:
        return f"No results found for query: {query}"
    
    except Exception as e:
        return f"Search error: {str(e)}"


# Tool definition for LLM
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for information. Use this tool to find current information, "
            "facts about people, recent events, or any topic. Returns a list of search "
            "results with title, snippet, and URL for each result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10, max: 20)",
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["query"]
        }
    }
}


if __name__ == "__main__":
    # Test the web search tool
    print("Testing web search tool...")
    
    test_queries = [
        "Python programming",
        "OpenAI GPT-4",
        "2024 Olympics"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        try:
            result = web_search(query, max_results=5)
            if result["success"]:
                print(f"Found {len(result['results'])} results:")
                for i, res in enumerate(result["results"], 1):
                    print(f"\n{i}. {res['title']}")
                    print(f"   URL: {res['url']}")
                    print(f"   Snippet: {res['snippet'][:100]}...")
            else:
                print(f"Error: {result['error']}")
        except Exception as e:
            print(f"Exception: {e}")

