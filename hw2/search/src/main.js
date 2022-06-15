import Vue from 'vue';
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
import App from './App.vue';
import TextHighlight from 'vue-text-highlight';


Vue.use(ElementUI);
Vue.component('text-highlight', TextHighlight);
new Vue({
  el: '#app',
  render: h => h(App)
});