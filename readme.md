## 环境
* 需预先安装node,vue,python3
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

