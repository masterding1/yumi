Page({
    data: {
      url: 'https://udify.app/chatbot/KlfHw2FwYooKBIPt'
    },
    onLoad: function (options) {
      // 如果有传入url参数，则使用传入的url
      if (options && options.url) {
        this.setData({
          url: decodeURIComponent(options.url)
        });
      }
    }
  })