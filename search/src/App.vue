<template>
  <el-container>
    <el-header>
    </el-header>

    <el-main>
      <li class="demo-input-suffix">

        <el-input placeholder="请输入内容" v-model="search_input" @keyup.enter.native="on_enter">
          <i slot="prefix" class="el-input__icon el-icon-search"></i>
        </el-input>
        <el-row :span="20" id="query_split">
          <el-col :span="3"><h3>搜索分词：</h3></el-col>
          <el-col :span="17">
            <el-input style="padding-top: 10px;" v-model="search_input_str" :disabled="true">
            </el-input>
          </el-col>
        </el-row>
      </li>
      <li v-for="(item, index) in items">
        <el-card class="box-card">
          <el-divider content-position="left">
            <h3>余弦距离</h3>
          </el-divider>
          {{item.cosine_distance}}
          <el-divider content-position="left">
            <h3>案件名称</h3>
          </el-divider>
          <text-highlight :queries="queries">{{ item.title }}</text-highlight>

          <el-divider content-position="left">
            <h3>事实</h3>
          </el-divider>
          <div v-if="item.fact.length > 500">
            <td v-if="!readMore[index]">
              <text-highlight :queries="queries">{{ item.fact.substring(0, 500) + ".." }}</text-highlight>
            </td>
            <td v-if="readMore[index]">
              <text-highlight :queries="queries">{{ item.fact }}</text-highlight>
            </td>
            <a @click="showMore(index)" v-if="!readMore[index]" class="show_more_less">更多</a>
            <a @click="showLess(index)" v-if="readMore[index]" class="show_more_less">更少</a>
          </div>
          <div v-else>
            <text-highlight :queries="queries">{{ item.fact }}</text-highlight>
          </div>

          <el-divider content-position="left">
            <h3>判决</h3>
          </el-divider>
          <div v-if="item.verdict.length > 500">
            <td v-if="!readMore[index]">
              <text-highlight :queries="queries">{{ item.verdict.substring(0, 500) + "..." }}</text-highlight>
            </td>
            <td v-if="readMore[index]">
              <text-highlight :queries="queries">{{ item.verdict }}</text-highlight>
            </td>
            <a @click="showMore(index)" v-if="!readMore[index]" class="show_more_less">更多</a>
            <a @click="showLess(index)" v-if="readMore[index]" class="show_more_less">更少</a>
          </div>
          <div v-else>
            <text-highlight :queries="queries">{{ item.verdict }}</text-highlight>
          </div>
        </el-card>
      </li>
    </el-main>
  </el-container>


</template>
<script>
import axios from 'axios';
const api = axios.create({
  baseURL: 'http://localhost:8000/',
});

export default {
  methods: {
    on_enter: function () {
      api.get("query", { "params": { "key": this.search_input } }).then(response => {
        this.queries = response.data.keys;
        this.items = response.data.result;
      }).catch(error => {
        console.log(error)
      });
      // console.log(this.axiosParams)
    },
    showMore(id) {
      this.$set(this.readMore, id, true);
    },
    showLess(id) {
      this.$set(this.readMore, id, false);
    }
  },
  computed: {
    search_input_str() {
      return this.queries.join(" | ");;
    }
  },
  data() {
    return {
      readMore: {},
      search_input: '',
      queries: [],
      items: []
    }
  }
}
</script>

<style>
/* #query_split {
  margin-top: 20px;
} */

li {
  list-style-type: none;
  padding-top: 25px;
  padding-bottom: 25px;
  padding-left: 400px;
  padding-right: 400px;

}

.text {
  font-size: 14px;
}

.item {
  padding: 18px 0;
}

.box-card {
  width: 960px;

}

.show_more_less {
  height: auto;
  overflow: visible;
  color: blue;
}
</style>