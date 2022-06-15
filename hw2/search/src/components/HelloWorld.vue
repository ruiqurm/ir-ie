Vue.component('child', {
  props: ['msg', 'search', 'effect'],
  template: '<span><span v-for="(s, i) in parsedMsg" v-bind:class="getClass(i%2)">{{s}}</span></span>',
  methods: {
    getClass: function(i) {
      var myClass = {};
      myClass[this.effect] = !!i;
      return myClass;
    },
  },
  computed: {
    parsedSearch : function () {
        return '(' + this.search.trim().replace(/ +/g, '|') + ')';
    },
    parsedMsg: function() {
        return this.msg.split(
        new RegExp(this.parsedSearch , 'gi'));
    }
  }
})
