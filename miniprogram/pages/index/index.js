// 在现有的 Page 对象中添加以下方法
Page({
  data: {
    // imgSrc: '', // 不需要在此页面直接使用imgSrc，选择后直接跳转
  },
  chooseImage: function(event) {
    const sourceType = event.currentTarget.dataset.sourcetype; // 获取是拍照还是相册
    wx.chooseImage({
      count: 1,
      sourceType: [sourceType], // 'album' 或 'camera'
      success: (res) => {
        const tempFilePath = res.tempFilePaths[0];
        wx.navigateTo({
          url: '/pages/preview/preview?imgSrc=' + tempFilePath
        });
      },
      fail: (err) => {
        console.error("选择图片失败:", err);
        if (err.errMsg !== "chooseImage:fail cancel") {
          wx.showToast({
            title: '选择图片失败',
            icon: 'none'
          });
        }
      }
    });
  },
  // chooseFromAlbum 函数已合并到 chooseImage 中通过 data-sourcetype 区分
  
  navigateToChat: function() {
    // 跳转到聊天页面
    wx.navigateTo({
      url: '/pages/webview/webview?url=' + encodeURIComponent('https://udify.app/chat/KlfHw2FwYooKBIPt')
    })
  },
  
  navigateToWebview: function() {
    // 跳转到内嵌网页页面
    wx.navigateTo({
      url: '/pages/webview/webview'
    })
  }
})