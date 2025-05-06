// preview.js
Page({
  data: {
    imgSrc: ''
  },
  onLoad: function(options) {
    if (options.imgSrc) {
      this.setData({
        imgSrc: options.imgSrc
      });
    } else {
      wx.showToast({ title: '图片路径错误', icon: 'none' });
      wx.navigateBack();
    }
  },
  uploadImage: function() {
    if (!this.data.imgSrc) {
      wx.showToast({ title: '未选择图片', icon: 'none' });
      return;
    }
    const localImgSrc = this.data.imgSrc;
    wx.showLoading({ title: '正在识别...' });
    wx.uploadFile({
      url: 'http://172.20.10.3:5000/predict', // 请确保这是您正确的服务器地址
      filePath: localImgSrc,
      name: 'image',
      // formData: { // 如果需要传递其他参数
      //   time: new Date().toLocaleString(),
      //   location: '未知', 
      //   growth_stage: '未知'
      // },
      success: (res) => {
        wx.hideLoading();
        try {
          const serverData = JSON.parse(res.data);
          if (serverData.success) {
            const navigationData = {
              ...serverData,
              imgSrc: localImgSrc // 将本地图片路径也传递给结果页
            };
            wx.navigateTo({
              url: '/pages/result/result?data=' + encodeURIComponent(JSON.stringify(navigationData))
            });
          } else {
            wx.showToast({
              title: serverData.error || '识别失败，请重试',
              icon: 'none',
              duration: 2000
            });
          }
        } catch (e) {
          console.error("解析服务器数据失败:", res.data, e);
          wx.showToast({
            title: '服务器响应格式错误',
            icon: 'none',
            duration: 2000
          });
        }
      },
      fail: (err) => {
        wx.hideLoading();
        console.error("上传失败:", err);
        wx.showToast({
          title: '上传识别失败，请检查网络',
          icon: 'none',
          duration: 2000
        });
      }
    });
  },
  goBack: function() {
    wx.navigateBack();
  }
})