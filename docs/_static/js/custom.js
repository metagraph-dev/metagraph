
const toggleIconSVGAnimation = (cardElement) => {
    Array.prototype.forEach.call(cardElement.getElementsByTagName('svg'), svg => {
	svg.classList.toggle('custom-icon-animate');
    });
};
