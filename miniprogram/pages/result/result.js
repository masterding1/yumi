Page({
  data: {
    imgSrc: '',
    disease: '',
    confidence: '',
    suggestion: '',
    inference_time: '',
    class_probabilities: {},
    displayTime: '' // 用于显示格式化的时间
  },
  onLoad: function(options) {
    try {
      const data = JSON.parse(decodeURIComponent(options.data));
      this.setData({
        imgSrc: data.imgSrc,
        disease: data.disease,
        confidence: data.confidence ? (data.confidence * 100).toFixed(2) : 'N/A',
        suggestion: data.suggestion,
        inference_time: data.inference_time,
        class_probabilities: data.class_probabilities,
        displayTime: new Date().toLocaleDateString() + " " + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) // 示例时间
      });
    } catch (e) {
      console.error("Error parsing data in result page:", e);
      wx.showToast({
        title: '加载结果失败',
        icon: 'none'
      });
    }
  },
  goHome: function() {
    wx.reLaunch({
      url: '/pages/index/index'
    });
  },
  goBack: function() {
    wx.navigateBack();
  }
});