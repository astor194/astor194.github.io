// 移动端导航菜单切换
const hamburger = document.querySelector('.hamburger');
const navMenu = document.querySelector('.nav-menu');

if (hamburger && navMenu) {
    hamburger.addEventListener('click', () => {
        navMenu.classList.toggle('active');
        hamburger.classList.toggle('active');
    });
}

// 搜索功能
const searchInput = document.querySelector('.hero-search input');
const searchButton = document.querySelector('.hero-search button');

if (searchButton) {
    searchButton.addEventListener('click', (e) => {
        e.preventDefault();
        const searchTerm = searchInput.value.trim();
        if (searchTerm) {
            console.log('搜索:', searchTerm);
            // 这里可以添加实际的搜索功能
            alert(`搜索: ${searchTerm}`);
        }
    });
}

if (searchInput) {
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            searchButton.click();
        }
    });
}

// 订阅表单
const subscribeForm = document.querySelector('.subscribe-form');

if (subscribeForm) {
    subscribeForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const emailInput = subscribeForm.querySelector('input[type="email"]');
        const email = emailInput.value.trim();
        
        if (email) {
            console.log('订阅邮箱:', email);
            alert('感谢订阅！我们会将最新文章发送到您的邮箱。');
            emailInput.value = '';
        }
    });
}

// 平滑滚动
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        if (href !== '#' && href.length > 1) {
            const target = document.querySelector(href);
            if (target) {
                e.preventDefault();
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        }
    });
});

// 滚动时导航栏阴影效果
const navbar = document.querySelector('.navbar');

window.addEventListener('scroll', () => {
    if (window.scrollY > 10) {
        navbar.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.boxShadow = '0 1px 2px 0 rgba(0, 0, 0, 0.05)';
    }
});

// 文章卡片动画
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

document.querySelectorAll('.post-card').forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    observer.observe(card);
});

// 移动端菜单样式调整
const mediaQuery = window.matchMedia('(max-width: 768px)');

function handleMobileMenu(e) {
    if (e.matches) {
        // 移动端样式
        if (navMenu) {
            navMenu.style.position = 'absolute';
            navMenu.style.top = '100%';
            navMenu.style.left = '0';
            navMenu.style.right = '0';
            navMenu.style.backgroundColor = 'white';
            navMenu.style.flexDirection = 'column';
            navMenu.style.padding = '1rem';
            navMenu.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1)';
            navMenu.style.display = 'none';
        }
    } else {
        // 桌面端样式
        if (navMenu) {
            navMenu.style.position = 'static';
            navMenu.style.flexDirection = 'row';
            navMenu.style.padding = '0';
            navMenu.style.boxShadow = 'none';
            navMenu.style.display = 'flex';
            navMenu.classList.remove('active');
        }
        if (hamburger) {
            hamburger.classList.remove('active');
        }
    }
}

// 初始检查
handleMobileMenu(mediaQuery);

// 监听变化
mediaQuery.addListener(handleMobileMenu);

// 汉堡菜单动画
if (hamburger) {
    const originalListener = hamburger.onclick;
    hamburger.onclick = function(e) {
        if (originalListener) originalListener.call(this, e);
        
        if (navMenu.style.display === 'none' || !navMenu.style.display) {
            navMenu.style.display = 'flex';
        } else {
            navMenu.style.display = 'none';
        }
    };
}

// 打印加载完成消息
console.log('博客主页加载完成！');

