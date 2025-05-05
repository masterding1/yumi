Component({
  options: {
    multipleSlots: true
  },
  properties: {
    title: {
      type: String,
      value: ''
    },
    background: {
      type: String,
      value: ''
    },
    color: {
      type: String,
      value: ''
    },
    back: {
      type: Boolean,
      value: true
    },
    loading: {
      type: Boolean,
      value: false
    },
    animated: {
      type: Boolean,
      value: true
    },
    show: {
      type: Boolean,
      value: true,
      observer: '_showChange'
    },
    delta: {
      type: Number,
      value: 1
    }
  },
  methods: {
    _showChange(show) {
      const animated = this.data.animated;
      if (animated) {
        this.setData({
          opacity: show ? 1 : 0
        });
      } else {
        this.setData({
          opacity: show ? 1 : 0
        });
      }
    },
    back() {
      const data = this.data;
      if (data.delta) {
        wx.navigateBack({
          delta: data.delta
        });
      }
    }
  }
})
