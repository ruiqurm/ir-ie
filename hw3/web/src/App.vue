<template>
<el-container>
<el-aside width="400px"></el-aside>
<el-main id="main">
  <el-row :gutter="20">
    <el-col :span="20">
        <el-input
        type="textarea"
        :autosize="{ minRows: 3, maxRows: 10}"
        placeholder="请输入内容"
        v-model="textarea">
        </el-input>
    </el-col>
    <el-col :span="4">
        <el-button type="primary" id="extract_button" v-on:click="on_click">抽取</el-button>
    </el-col>
  </el-row>
  <el-row id ="selection">
    <el-checkbox-group v-model="checkList" :max="16">
    <span v-for="obj in features" :key="obj.feature_id" class="box">
        <el-checkbox :label="obj.feature_id">{{obj.feature_text}}</el-checkbox>
    </span>
    </el-checkbox-group>
  </el-row>
  <el-divider content-position="center"></el-divider>
  <el-table
      :data="tableData">
      <el-table-column
        prop="feature"
        label="信息点"
        width="180">
      </el-table-column>
      <el-table-column
        prop="pos"
        label="预测位置"
        width="180">
      </el-table-column>
      <el-table-column
        prop="content"
        label="内容"
        width="180">
      </el-table-column>
    </el-table>
  <el-divider content-position="center"></el-divider>
  <el-card class="box-card" v-html="result">
  </el-card>
</el-main>
<el-aside width="400px"></el-aside>
</el-container>
</template>

<script>
import axios from 'axios'
const api = axios.create({
  baseURL: 'http://localhost:8000/'
})
export default {
  methods: {
    on_click: function () {
      if (this.checkList.length === 0) {
        return
      }

      api.post('query', {text: this.textarea, feature_ids: this.checkList}).then((res) => {
        let result = ''
        let curpos = 0
        for (var item of res.data.intervals) {
          let left = this.textarea.substr(curpos, item[0] - curpos)
          result = result + left + '<span class="highlight">' + this.textarea.substr(item[0], item[1] - item[0]) + '</span>'
          curpos = item[1]
        }
        if (curpos < this.textarea.length) {
          let remain = this.textarea.substr(curpos)
          result = result + remain
        }
        let sl = []
        for (var obj of res.data.result) {
          let s = ''
          for (var tu of obj['pos']) {
            s += '(' + tu.toString() + ')  '
          }
          sl.push({feature: obj['feature'], pos: s, content: obj['content']})
        }
        this.tableData = sl
        this.result = result
      }).catch((err) => {
        console.log(err)
      })
    }
  },
  data () {
    return {
      textarea: '',
      result: '',
      checkList: [],
      features: [],
      tableData: []
    }
  },
  mounted () {
    api.get('/feature').then((result) => {
      this.features = result.data
    }).catch(function (error) {
      console.log(error)
    })
  }
}
</script>

<style>
.highlight{
    color: red;
}
.box{
    margin-right: 25px;

}
/* #app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
} */
#extract_button{
    margin-top: 15px;
}
#selection{
    padding-top: 50px;
    padding-bottom: 50px;
}
.box-card{
    height: 500px;
}
</style>
