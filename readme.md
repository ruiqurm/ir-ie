## 环境
* 需预先安装node,vue,python3
* 下载数据：https://drive.google.com/file/d/1vQdX1MegFVtmoh0XCd4mav5PBkep7q0h/view
* 解压数据包到data文件夹下(选择解压出documents)
```
pip install -r requirements.txt
cd search
npm install
```



## 运行

生成倒排索引

```
python build.py
```

启动服务器
```
uvicorn main:app --reload
npm run dev
```

在localhost:8080下面启动网页

