Page({
  data: {
    imgSrc: '',
    disease: '',
    confidence: '',
    // suggestion: '', // 旧的建议字段，将被替换
    inference_time: '',
    class_probabilities: {},
    displayTime: '', // 用于显示格式化的时间
    // 新增字段，用于接收详细信息
    symptoms: '', // 症状
    cause: '',    // 病因
    harm: '',     // 危害
    prevention_advice: '', // 防治建议
    location_info: '', // 地点信息
    time_info: ''      // 时间信息
  },
  onLoad: function(options) {
    try {
      const data = JSON.parse(decodeURIComponent(options.data));
      this.setData({
        imgSrc: data.imgSrc,
        disease: data.disease,
        // 注意：这里的 confidence 计算依赖服务器返回一个 0-1 之间的概率值。
        // 如果服务器返回的 data.confidence 已经是百分比或者是一个较大的数（如截图中的情况，data.confidence 可能约为 7.67），
        // 则此处的 * 100 会导致显示异常。请确保服务器返回正确的原始概率。
        confidence: data.confidence ? (data.confidence * 100).toFixed(2) : 'N/A',
        // suggestion: data.suggestion, // 移除旧的建议
        inference_time: data.inference_time,
        class_probabilities: data.class_probabilities,
        displayTime: new Date().toLocaleDateString() + " " + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), // 示例时间
        // 设置新的详细信息字段
        symptoms: data.symptoms || '暂无信息',
        cause: data.cause || '暂无信息',
        harm: data.harm || '暂无信息',
        prevention_advice: data.prevention_advice || '暂无信息',
        location_info: data.location_info || '未知',
        time_info: data.time_info || '未知'
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