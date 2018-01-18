import ssl
import bs4
import re
import requests
import csv
import codecs
import time

from urllib import request, error

context = ssl._create_unverified_context()


class DouBanSpider:
    def __init__(self):
        self.userAgent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"
        self.headers = {"User-Agent": self.userAgent}

    # 拿到豆瓣图书的分类标签
    def getBookCategroies(self):
        try:
            url = "https://book.douban.com/tag/?view=type&icn=index-sorttags-all"
            response = request.urlopen(url, context=context)
            content = response.read().decode("utf-8")
            return content
        except error.HTTPError as identifier:
            print("errorCode: " + identifier.code + "errrorReason: " + identifier.reason)
            return None

    # 找到每个标签的内容
    def getCategroiesContent(self):
        content = self.getBookCategroies()
        if not content:
            print("页面抓取失败...")
            return None
        soup = bs4.BeautifulSoup(content, "lxml")
        categroyMatch = re.compile(r"^/tag/*")
        categroies = []
        for categroy in soup.find_all("a", {"href": categroyMatch}):
            if categroy:
                categroies.append(categroy.string)
        return categroies

    # 拿到每个标签的链接
    def getCategroyLink(self):
        categroies = self.getCategroiesContent()
        categroyLinks = []
        for item in categroies:
            link = "https://book.douban.com/tag/" + str(item)
            categroyLinks.append(link)
        return categroyLinks

    def getBookInfo(self, categroyLinks):
        self.setCsvTitle()
        categroies = categroyLinks
        try:
            for link in categroies:
                print("正在爬取：" + link)
                bookList = []
                response = requests.get(link)
                soup = bs4.BeautifulSoup(response.text, 'lxml')
                bookCategroy = soup.h1.string
                for book in soup.find_all("li", {"class": "subject-item"}):
                    bookSoup = bs4.BeautifulSoup(str(book), "lxml")
                    bookTitle = bookSoup.h2.a["title"]
                    bookAuthor = bookSoup.find("div", {"class": "pub"})
                    bookComment = bookSoup.find("span", {"class": "pl"})
                    bookContent = bookSoup.li.p
                    # print(bookContent)
                    if bookTitle and bookAuthor and bookComment and bookContent:
                        bookList.append([bookTitle.strip(),bookCategroy.strip() , bookAuthor.string.strip(),
                                         bookComment.string.strip(), bookContent.string.strip()])
                self.saveBookInfo(bookList)
                time.sleep(3)

            print("爬取结束....")

        except error.HTTPError as identifier:
            print("errorCode: " + identifier.code + "errrorReason: " + identifier.reason)
            return None

    def setCsvTitle(self):
        csvFile = codecs.open("data/data.csv", 'a', 'utf_8_sig')
        try:
            writer = csv.writer(csvFile)
            writer.writerow(['title', 'tag', 'info', 'comments', 'content'])
        finally:
            csvFile.close()

    def saveBookInfo(self, bookList):
        bookList = bookList
        csvFile = codecs.open("data/data.csv", 'a', 'utf_8_sig')
        try:
            writer = csv.writer(csvFile)
            for book in bookList:
                writer.writerow(book)
        finally:
            csvFile.close()

    def start(self):
        categroyLink = self.getCategroyLink()
        self.getBookInfo(categroyLink)


douBanSpider = DouBanSpider()
douBanSpider.start()
