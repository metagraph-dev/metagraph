
const toggleIconSVGAnimation = (cardElement) => {
    Array.prototype.forEach.call(cardElement.getElementsByTagName('svg'), svg => {
	svg.classList.add('custom-icon-animate');
    });
};

const makeCustomButtonsMatchChildLinks = () => {
    Array.prototype.forEach.call(document.getElementsByClassName('custom-button'), customButton => {
        const anchors = customButton.getElementsByTagName('a');
        if (anchors.length != 1) {
            console.warn(`${customButton} got an unexpected number of child anchors.`);
        } else {
            const anchor = anchors[0];
            const href = anchor.getAttribute('href');
            const wrappingAnchor = document.createElement('a');
            wrappingAnchor.setAttribute('href', href);
            customButton.parentNode.insertBefore(wrappingAnchor, customButton);
            wrappingAnchor.appendChild(customButton);
        }
    });
};
